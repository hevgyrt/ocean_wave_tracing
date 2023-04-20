import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xa
import pyproj
import sys
import cmocean.cm as cm
from netCDF4 import Dataset
import json
from importlib import resources

from .util_solvers import Advection, WaveNumberEvolution, RungeKutta4
from .util_methods import make_xarray_dataArray, to_xarray_ds, check_velocity_field, check_bathymetry


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ocean_wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')


class Wave_tracing():
    """ Class computing the path of ocean wave rays according to the geometrical
    optics approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 temporal_evolution=False,
                 d=None,DEBUG=False,**kwargs):
        """
        Args:
            U (float): eastward velocity 2D field
            V (float): northward velocity 2D field
            nx (int): number of points in x-direction of velocity domain
            ny (int): number of points in y-direction of velocity domain
            nt (int): number of time steps for computation
            T (int): Seconds. Duration of wave tracing
            dx (int): Spatial resolution in x-direction. Units conforming to U
            dy (int): Spatial resolution in y-direction. Units conforming to V
            nb_wave_rays (int): Number of wave rays to track.
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction
            temporal_evolution (bool): flag if velocity field should change in time
            d (float): 2D bathymetry field
            **kwargs
        """
        self.g = 9.81
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.nb_wave_rays = nb_wave_rays
        self.C = -1
        assert nb_wave_rays > 0, "Number of wave rays must be larger than zero"

        self.domain_X0 = domain_X0 # left side
        self.domain_XN = domain_XN # right side
        self.domain_Y0 = domain_Y0 # bottom
        self.domain_YN = domain_YN # top
        self.T = T

        self.temporal_evolution = temporal_evolution
        self.debug = DEBUG

        # Setting up X and Y domain
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        # Check the bathymetry
        if d is not None:
            self.d = check_bathymetry(d=d,x=self.x,y=self.y)
        else:
            d_static = 1e5
            logging.info(f'Hardcoding bathymetry to {d_static}m since not given.')
            self.d = check_bathymetry(d=np.ones((ny,nx))*d_static,x=self.x,y=self.y)


        # Setting up the wave rays
        self.ray_x = np.zeros((nb_wave_rays,nt))
        self.ray_y = np.zeros((nb_wave_rays,nt))
        self.ray_kx = np.zeros((nb_wave_rays,nt))
        self.ray_ky = np.zeros((nb_wave_rays,nt))
        self.ray_k = np.zeros((nb_wave_rays,nt))
        self.ray_theta = np.ma.zeros((nb_wave_rays,nt))
        self.ray_cg = np.ma.zeros((nb_wave_rays,nt)) # intrinsic group velocity
        self.ray_U = np.ma.zeros((nb_wave_rays,nt)) # U component closest to ray
        self.ray_V = np.ma.zeros((nb_wave_rays,nt)) # V component closest to ray
        self.ray_depth = np.zeros((nb_wave_rays,nt))

        # bathymetry gradient
        self.dsigma_dx = np.ma.zeros((nb_wave_rays,nt))
        self.dsigma_dy = np.ma.zeros((nb_wave_rays,nt))
        
        # make xarray DataArray of velocity field
        self.U = check_velocity_field(U,temporal_evolution,x=self.x,y=self.y)
        self.V = check_velocity_field(V,temporal_evolution,x=self.x,y=self.y)

        # Time
        self.dt = T/nt
        self.nb_velocity_time_steps = len(self.U.time)
        if not temporal_evolution:
            self.velocity_idt = np.zeros(nt,dtype=int)
            #logging.info('Vel idt : {}'.format(self.velocity_idt))
        else:
            t_velocity_field = U.time.data
            self.T0 = t_velocity_field[0]
            t_wr = np.arange(self.T0, self.T0+np.timedelta64(T,'s'),np.timedelta64(int(((T/nt)*1e3)),'ms'))
            self.velocity_idt = np.array([self.find_nearest(t_velocity_field,t_wr[i]) for i in range(len(t_wr))])

        self.kwargs = kwargs


    def check_CFL(self, cg, max_speed):
        """ Method for checking the Courant, Friedrichs, and Lewy
            condition for numerical intergration
        """
        dt = self.dt
        DX = np.min([self.dx,self.dy])

        assert cg>=0, "Group velocity must be positive. Currently {}".format(cg)
        assert max_speed>=0,  "Maximum current speed must be positive. Currently {}".format(max_speed)

        u = cg+max_speed

        C = u*(dt/DX)

        if C<=1:
            logger.info('Courant number is {}'.format(np.round(C,2)))
        else:
            logger.warning('Courant number is {}'.format(np.round(C,2)))

        self.C = C


    def find_nearest(self,array, value):
        """ Method finding nearest indices to a position in array

        Args:
            array: Array containg to be compared with the value
            value (float): value to which index in array should be found

        Returns:
            idx (int): Index of the array which is closest to the value
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    def c_intrinsic(self,k,d,group_velocity=False):
        """ Computing the intrinsic wave phase and group velocity according
        to the general dispersion relation

        Args:
            k (float): wave number (numpy array)
            d (float): depth (numpy array)
            group_velocity (bool): returns group velocity (True) or phase
                velocity (False)

        Returns:
            instrinsic velocity (float): group or phase velocity depending on
                flag
        """

        g=self.g
        dw_criteria = k*d>25

        if dw_criteria.all():
            c_in = np.sqrt(g/k)
            n=0.5
        else:
            c_in = np.sqrt((g/k)*np.tanh(k*d)) #intrinsic
            n = 0.5 * (1 + (2*k*d)/np.sinh(2*k*d))

        if group_velocity:
            return c_in*n
        else:
            return c_in

    def sigma(self,k,d):
        """ Intrinsic frequency dispersion relation

        Args:
            k (float): Wave number
            d (float): depth

        Returns:
            sigma (float): intrinsic frequency
        """

        g=self.g
        sigma = np.sqrt(g*k*np.tanh(k*d))
        return sigma

    def dsigma(self,k,idxs,idys,dx, direction):
        """ Compute the horizontal gradient in sigma due to
        the bathymetry using a central difference scheme.
        """

        # Fixing indices outside domain
        idxs[idxs<1] = 1
        idxs[idxs>=self.nx-1] = self.nx-2
        idys[idys<1] = 1
        idys[idys>=self.ny-1] = self.ny-2


        if direction == 'x':
            ray_depth_last = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs-1,dims='z'))
            ray_depth_next = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs+1,dims='z'))
        elif direction == 'y':
            ray_depth_last = self.d.isel(y=xa.DataArray(idys-1,dims='z'),x=xa.DataArray(idxs,dims='z'))
            ray_depth_next = self.d.isel(y=xa.DataArray(idys+1,dims='z'),x=xa.DataArray(idxs,dims='z'))

        dsigma = (1/(2*dx)) * (self.sigma(k,ray_depth_next) - self.sigma(k,ray_depth_last))

        return dsigma


    def wave(self,T,theta,d):
        """ Method computing wave number from initial wave period

        Args:
            T (float): Wave period
            theta (float): radians. Wave direction

        Returns:
            k0 (float): wave number
            ray_kx0 (float): wave number in x-direction
            ray_ky0 (float): wave number in y-direction
        """
        g=self.g

        sigma = (2*np.pi)/T
        k0 = (sigma**2)/g

        # approximation of wave number according to Eckart (1952)
        alpha = k0*d
        k = (alpha/np.sqrt(np.tanh(alpha)))/d

        kx = k*np.cos(theta)
        ky = k*np.sin(theta)
        #logger.info('wave: {}, {},{}'.format(k,kx,ky))
        return k,kx,ky


    def set_initial_condition(self, wave_period, theta0,**kwargs):
        """ Setting inital conditions for the domain. Support domain side and
            initial x- and y-positions. However, side trumps initial position
            if both are given.

            args:
            wave_period (float): Wave period.
            theta0 (rad, float): Wave initial direction. In radians.
                         (0,.5*pi,pi,1.5*pi) correspond to going
                         (right, up, left, down).

            **kwargs
                incoming_wave_side (str): side for incoming wave direction
                                            [left, right, top, bottom]
                ipx (float, array of floats): initial position x
                ipy (float, array of floats): initial position y
        """

        nb_wave_rays = self.nb_wave_rays

        valid_sides = ['left', 'right','top','bottom']

        if 'incoming_wave_side' in kwargs:
            i_w_side = kwargs['incoming_wave_side']

            if not i_w_side in valid_sides:
                logger.info('Invalid initial side. Left will be used.')
                i_w_side = 'left'

            if i_w_side == 'left':
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'right':
                xs = np.ones(nb_wave_rays)*self.domain_XN
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'top':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_YN

            elif i_w_side == 'bottom':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_Y0

        else:
            logger.info('No initial side given. Try with discrete points')

            ipx = kwargs.get('ipx', None)
            ipy = kwargs.get('ipy', None)

            if ipx is None and ipy is None:
                logger.info('No initial position points given. Left will be used')
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            else:
                # First check initial position x
                if np.isfinite(ipx).all():
                    #Check if it is an array
                    if isinstance(ipx,np.ndarray):
                        assert nb_wave_rays == len(kwargs['ipx']), "Need same dimension on initial x-values"
                        xs=ipx.copy()
                    # if not, use it as single value
                    else:
                        ipx = np.ones(nb_wave_rays)*ipx
                        xs=ipx.copy()

                if np.isfinite(ipy).all():
                    if isinstance(ipy,np.ndarray):
                        assert nb_wave_rays == len(kwargs['ipy']), "Need same dimension on initial x-values"
                        ys=ipy.copy()
                    else:
                        ipy = np.ones(nb_wave_rays)*ipy
                        ys=ipy.copy()


        # Set initial position
        self.ray_x[:,0] = xs
        self.ray_y[:,0] = ys

        #Theta0
        if type(theta0) is float or type(theta0) is int:
            theta0 = np.ones(nb_wave_rays)*theta0
        elif isinstance(theta0,np.ndarray):
            assert nb_wave_rays == len(theta0), "Initial values must have same dimension as number of wave rays"
        else:
            logger.error('Theta0 must be either float or numpy array. Terminating.')
            sys.exit()


        # set inital wave properties
        for i in range(nb_wave_rays):
            self.ray_k[i,0], self.ray_kx[i,0], self.ray_ky[i,0] = self.wave(T=wave_period,
                                                                theta=theta0[i],
                                                                d=self.d.sel(y=ys[i],x=xs[i],method='nearest'))
            self.ray_cg[i,0] = self.c_intrinsic(k=self.ray_k[i,0],d=self.d.sel(y=ys[i],x=xs[i],method='nearest'),group_velocity=True)

        # set inital wave propagation direction
        self.ray_theta[:,0] = theta0

        #Check the CFL condition
        self.check_CFL(cg=np.nanmax(self.ray_cg[:,0]),max_speed=np.nanmax(np.sqrt(self.U**2+self.V**2)))


    def solve(self, solver=RungeKutta4):
        """ Solve the geometrical optics equations numerically by means of the
            method of characteristics
        """

        if not callable(solver):
            raise TypeError('f is %s, not a solver' % type(solver))

        ray_k = self.ray_k
        ray_kx= self.ray_kx
        ray_ky= self.ray_ky
        ray_x= self.ray_x
        ray_y= self.ray_y
        ray_theta= self.ray_theta
        ray_cg = self.ray_cg
        ray_U = self.ray_U
        ray_V = self.ray_V

        dx = self.dx
        dy = self.dy

        U = self.U.data
        V = self.V.data

        #Compute velocity gradients
        logger.info('Assuming uniform horizontal resolution in each direction')
        if self.nb_velocity_time_steps>1:
            dudy, dudx = np.gradient(U,dx)[1:3]
            dvdy, dvdx = np.gradient(V,dy)[1:3]
        else:
            dudy, dudx = np.gradient(U[0,:,:],dx)
            dvdy, dvdx = np.gradient(V[0,:,:],dy)

            dudy = np.expand_dims(dudy,axis=0)
            dudx = np.expand_dims(dudx,axis=0)
            dvdy = np.expand_dims(dvdy,axis=0)
            dvdx = np.expand_dims(dvdx,axis=0)

        x = self.x
        y = self.y
        dt = self.dt
        nt = self.nt
        velocity_idt = self.velocity_idt


        counter=0
        t = np.linspace(0,self.T,nt)

        for n in range(0,nt-1):

            # find indices for each wave ray
            idxs = np.array([self.find_nearest(x,xval) for xval in ray_x[:,n]])
            idys = np.array([self.find_nearest(y,yval) for yval in ray_y[:,n]])

            ray_depth = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs,dims='z'))
            self.ray_depth[:,n] = ray_depth

            self.ray_U[:,n] = self.U.isel(time=velocity_idt[n], y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs,dims='z'))
            self.ray_V[:,n] = self.V.isel(time=velocity_idt[n], y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs,dims='z'))

            #======================================================
            ### numerical integration of the wave ray equations ###
            #======================================================

            # Compute group velocity
            ray_cg[:,n+1] = self.c_intrinsic(ray_k[:,n],d=ray_depth,group_velocity=True)

            # ADVECTION
            f_adv = Advection(cg=ray_cg[:,n+1], k=ray_k[:,n], kx=ray_kx[:,n], U=U[velocity_idt[n],idys,idxs])
            ray_x[:,n+1] = solver.advance(u=ray_x[:,n], f=f_adv,k=n,t=t) # NOTE: this k is a counter and not wave number

            f_adv = Advection(cg=ray_cg[:,n+1], k=ray_k[:,n], kx=ray_ky[:,n], U=V[velocity_idt[n],idys,idxs])
            ray_y[:,n+1] = solver.advance(u=ray_y[:,n], f=f_adv, k=n, t=t)# NOTE: this k is a counter and not wave number


            # EVOLUTION IN WAVE NUMBER
            self.dsigma_dx[:,n+1] = self.dsigma(ray_k[:,n], idxs, idys, self.dx,direction='x')
            self.dsigma_dy[:,n+1] = self.dsigma(ray_k[:,n], idxs, idys, self.dx,direction='y')

            f_wave_nb = WaveNumberEvolution(d_sigma=self.dsigma_dx[:,n+1], kx=ray_kx[:,n], ky=ray_ky[:,n],
                                               dUkx=dudx[velocity_idt[n],idys,idxs], dUky=dvdx[velocity_idt[n],idys,idxs])
            ray_kx[:,n+1] = solver.advance(u=ray_kx[:,n], f=f_wave_nb,k=n, t=t)# NOTE: this k is a counter and not wave number

            f_wave_nb = WaveNumberEvolution(d_sigma=self.dsigma_dy[:,n+1], kx=ray_kx[:,n], ky=ray_ky[:,n],
                                               dUkx=dudy[velocity_idt[n],idys,idxs], dUky=dvdy[velocity_idt[n],idys,idxs])
            ray_ky[:,n+1] = solver.advance(u=ray_ky[:,n], f=f_wave_nb, k=n, t=t)# NOTE: this k is a counter and not wave number

            # Compute wave number k
            ray_k[:,n+1] = np.sqrt(ray_kx[:,n+1]**2+ray_ky[:,n+1]**2)

            # THETA
            ray_theta[:,n+1] = np.arctan2(ray_ky[:,n+1],ray_kx[:,n+1])
            #logging.info(ray_theta[:,n+1])
            #keep angles between 0 and 2pi
            #ray_theta[:,n+1] = ray_theta[:,n+1]%(2*np.pi)
            ray_theta[:,n+1] = np.mod(ray_theta[:,n+1],(2*np.pi))


            counter += 1

            # Logging purposes
            if self.debug:
                #logging.info("idxs:{}".format(idxs))
                #logging.info("idys:{}".format(idxs))
                #logging.info("dsigma_ds:{}".format(dsigma_ds[wr_id]))
                #logging.info("phi: {}".format(np.cos(phi[wr_id])))
                if counter in range(1200,1300,250):
                    #wr_id = [20, 40, 105, 80, 130, 150, 170] # wave ray ID for illustration in idealized_input
                    wr_id = [20, 40, 80,105] # wave ray ID

                    #break

                    idts = np.arange(200,1000,200)
                    fs=12
                    fig3,ax3 = plt.subplots(nrows=4,ncols=1,figsize=(16,10),gridspec_kw={'height_ratios': [3, 1,1,1]})
                    pc=ax3[0].contourf(self.x,self.y,-self.d,shading='auto',cmap=cm.deep_r,levels=25)
                    for id in wr_id:
                        ax3[0].plot(self.ray_x[id,:n+1],self.ray_y[id,:n+1],'-k')

                    ax3[0].plot(wt.ray_x[wr_id[2],idts],wt.ray_y[wr_id[2],idts],marker='s',ms=7,color='tab:red',linestyle='none')

                    ax3[0].xaxis.tick_top()

                    ax3[1].plot(-wt.ray_depth[wr_id[2],:1090], label=r'$d(x_r,y_r)$')
                    ax3[2].plot(wt.ray_kx[wr_id[2],:1090], label=r'$k_x$')
                    ax3[2].plot(wt.ray_ky[wr_id[2],:1090], label=r'$k_y$',c='tab:green')
                    ax3[3].plot(wt.ray_theta[wr_id[2],:1090], label=r'$\theta$')

                    ax3[2].sharex(ax3[1])
                    ax3[3].sharex(ax3[1])

                    cb3 = fig3.colorbar(pc,ax=ax3[0])
                    cb3.ax.tick_params(labelsize=fs)

                    for ii, aax in enumerate([ax3[1],ax3[2],ax3[3]]):
                        for idt in idts:
                            aax.axvline(idt,c='tab:red',lw=1.5)
                        aax.grid()
                        aax.legend(fontsize=fs+4)

                    for aax in ax3:
                        aax.tick_params(labelsize=fs)

                    ax3[0].set_xlim([self.x[0],self.x[-1]])
                    ax3[0].set_ylim([self.y[0],self.y[-1]])
                    fig3.tight_layout()
                    #plt.show()


        self.dudy = dudy
        self.dudx = dudx
        self.dvdy = dvdy
        self.dvdx = dvdx
        self.ray_k = ray_k
        self.ray_kx= ray_kx
        self.ray_ky= ray_ky
        self.ray_x= ray_x
        self.ray_y= ray_y
        self.ray_theta = ray_theta
        self.ray_cg = ray_cg
        logging.info('Stoppet at time idt: {}'.format(velocity_idt[n]))

    def to_ds(self,**kwargs):
        """Convert wave ray information to xarray object"""

        if 'proj4' in kwargs:
            lons,lats = self.to_latlon(kwargs['proj4'])
        else:
            lons = np.zeros((self.nb_wave_rays,self.nt))
            lats = lons.copy()

        variables = {'ray_k':self.ray_k,
                    'ray_kx':self.ray_kx,
                    'ray_ky':self.ray_ky,
                    'ray_x':self.ray_x,
                    'ray_y':self.ray_y,
                    'ray_U':self.ray_U,
                    'ray_V':self.ray_V,
                    'ray_theta':self.ray_theta,
                    'ray_cg':self.ray_cg,
                    'ray_depth':self.ray_depth,
                    'ray_lat': lats,
                    'ray_lon':lons
                    }

        with resources.open_text('ocean_wave_tracing','ray_metadata.json') as f:
            data = json.load(f)

        # relative time
        t = np.linspace(0,self.T,self.nt)
        ray_id = np.arange(self.nb_wave_rays)

        vars = [make_xarray_dataArray(var=variables[vname], t=t,rays=ray_id,name=vname,attribs=data[vname]) for vname in list(variables.keys())]

        return to_xarray_ds(vars)

    def ray_density(self,x_increment, y_increment, plot=False):
        """ Method computing ray density within boxes. The density of wave rays
        can be used as proxy for wave energy density

        Args:
            x_increment (int): size of box in x direction. Length = x_increment*dx
            y_increment (int): size of box in y direction. Length = y_increment*dy

        Returns:
            xx (2d): x grid
            yy (2d): y grid
            hm (2d): heat map of wave ray density

        #>>> wt = Wave_tracing(U=1,V=1,nx=1,ny=1,nt=1,T=1,dx=1,dy=1,wave_period=1, theta0=1,nb_wave_rays=1,domain_X0=0,domain_XN=0,domain_Y0=0,domain_YN=1,incoming_wave_side='left')
        #>>> wt.solve()
        #>>> wt.ray_density(x_increment=20,y_increment=20)
        """
        xx,yy=np.meshgrid(self.x[::x_increment],self.y[::y_increment])
        hm = np.zeros(xx.shape) # heatmap
        xs = xx[0]
        ys = yy[:,0]

        counter=0
        for i in range(0,self.nb_wave_rays):
            for idx in range(len(xs)-1):
                x0, xn = xs[idx],xs[idx+1]
                for idy in range(len(ys)-1):
                    y0, yn = ys[idy],ys[idy+1]
                    counter+=1
                    valid_x = (self.ray_x[i,:]>x0)*(self.ray_x[i,:]<xn)
                    if (np.any((self.ray_y[i,:][valid_x]>y0)*(self.ray_y[i,:][valid_x]<yn))):
                        hm[idy,idx]+=1

        if plot:
            plt.pcolormesh(xx,yy,hm)
            plt.colorbar()
            for i in range(0,self.nb_wave_rays):
                plt.plot(self.ray_x[i,:],self.ray_y[i,:],'-r',alpha=0.3)
            plt.scatter(xx,yy);plt.show()

        return xx,yy,hm

    def to_latlon(self, proj4):
        """ Method for reprojecting wave rays to latitude/longitude values

        Args:
            proj4 (str): proj4 string

        Returns:
            lons (2d): wave rays in longitude
            lats (2d): wave rays in latitude
        """
        lats = np.zeros((self.nb_wave_rays,self.nt))
        lons = np.zeros((self.nb_wave_rays,self.nt))
        #print(pyproj.__dict__.keys())
        for i in range(self.nb_wave_rays):
            lons[i,:],lats[i,:] = pyproj.Transformer.from_proj(proj4,'epsg:4326', always_xy=True).transform(self.ray_x[i,:], self.ray_y[i,:])

        return lons, lats
