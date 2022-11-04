import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xa
import pyproj
import sys
import cmocean.cm as cm
from netCDF4 import Dataset
import json

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

import util_solvers as uts
from util_methods import make_xarray_dataArray


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ocean_wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')


class Wave_tracing():
    """ Class computing the path of ocean wave rays according to the geometrical
    optics approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 temporal_evolution=False, T0=None,
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
            nb_wave_rays (int): Number of wave rays to track. NOTE: Should be
                                equal or less to either nx or ny.
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction


            temporal_evolution (bool): flag if velocity field should change in time
            T0 (int): Initial time if temporal_evolution==True
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
        assert nb_wave_rays > 0, "Number of wave rays must be larger than zero"

        self.domain_X0 = domain_X0 # left side
        self.domain_XN = domain_XN # right side
        self.domain_Y0 = domain_Y0 # bottom
        self.domain_YN = domain_YN # top
        #self.i_w_side = incoming_wave_side
        self.T = T

        self.temporal_evolution = temporal_evolution
        self.debug = DEBUG

        # Setting up X and Y domain
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        if d is not None:
            self.d = self.check_bathymetry(d)
        else:
            logging.warning('Hardcoding bathymetry if not given. Should be fixed')
            self.d = self.check_bathymetry(np.ones((ny,nx))*1e9)
        self.ray_depth = np.zeros((nb_wave_rays,nt))


        # Setting up the wave rays
        self.xr = np.zeros((nb_wave_rays,nt))
        self.yr = np.zeros((nb_wave_rays,nt))
        self.kx = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.ky = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.k = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.theta = np.ma.zeros((nb_wave_rays,nt))
        self.cg_i = np.ma.zeros((nb_wave_rays,nt)) #intrinsic group velocity

        # bathymetry gradient
        self.dsigma_dx = np.ma.zeros((nb_wave_rays,nt))
        self.dsigma_dy = np.ma.zeros((nb_wave_rays,nt))
        # make xarray data array of velocity field

        if not type(U) == xa.DataArray:
            self.U = xa.DataArray(U)
            self.V = xa.DataArray(V)
            if not temporal_evolution:
                self.U = self.U.expand_dims('time')
                self.V = self.V.expand_dims('time')
        else:
            self.U = U
            self.V = V
            if not 'time' in self.U.dims:
                self.U = self.U.expand_dims('time')
                self.V = self.V.expand_dims('time')

        # Time
        self.dt = T/nt
        self.nb_velocity_time_steps = len(self.U.time)
        if not temporal_evolution:
            self.velocity_idt = np.zeros(nt,dtype=int)
            logging.info('Vel idt : {}'.format(self.velocity_idt))
        else:
            t_velocity_field = U.time.data
            self.T0 = t_velocity_field[0]
            t_wr = np.arange(self.T0, self.T0+np.timedelta64(T,'s'),np.timedelta64(int(((T/nt)*1e3)),'ms'))
            self.velocity_idt = np.array([self.find_nearest(t_velocity_field,t_wr[i]) for i in range(len(t_wr))])
            logging.info('Vel idt : {}'.format(self.velocity_idt))
            logging.info('length: {}, {}'.format(len(self.velocity_idt),nt))

        self.kwargs = kwargs

    def check_bathymetry(self,d):
        """ Method checking and fixing bathymetry input

        Args:
            d (float): 2d bathymetry field

        Returns:
            d (float): 2d xarray DataArray object
        """

        if np.any(d < 0) and np.any(d > 0):
            logger.warning('Depth is defined as positive. Thus, negative depth will be treated as Land.')
            d[d<0] = 0
            return d

        if np.any(d < 0):
            logger.warning('Depth is defined as positive. Hence taking absolute value of input.')
            d = np.abs(d)

        d[d==0] = np.nan

        if not type(d) == xa.DataArray:
            d = xa.DataArray(data=d,
                             dims=['y','x'],
                             coords=dict(
                                    x=(['x'], self.x),
                                    y=(['y'], self.y),
                                    )
                            )
        return(d)

    def find_nearest(self,array, value):
        """ Method for finding neares indicies to value in array

        Args:
            array: Array containg values to be compared
            value (float): value to which index in array should be found

        Returns:
            idx (int): Index of array value closest to value
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    def c_intrinsic(self,k,d=None,group_velocity=False):
        """ Method computing intrinsic wave phase and group velocity according
        to the general dispersion relation

        Args:
            k: wave number
            group_velocity (bool): returns group velocity (True) or phase
                velocity (False)

        Returns:
            instrinsic velocity (float): group or phase velocity depending on
                flag
        """

        g=self.g

        if d is None:
            c_in = np.sqrt(g/k)
            n=0.5
        else:
            c_in = np.sqrt((g/k)*np.tanh(k*d)) #intrinsic
            n = 0.5 * (1 + (2*k*d)/np.sinh(2*k*d))

        if group_velocity:
            return c_in*n
            #return 0.5*np.sqrt(g/k)
        else:
            return c_in
            #return np.sqrt(g/k)

    def sigma(self,k,d):
        """ frequency dispersion relation

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

        # Fixing indices outside domain
        idxs[idxs<1] = 1
        idxs[idxs>=self.nx-1] = self.nx-2
        idys[idys<1] = 1
        idys[idys>=self.ny-1] = self.ny-2


        if direction == 'x':
            ray_depth_last = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs-1,dims='z'))
            ray_depth_next = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs+1,dims='z'))
            #dsigma = (1/(2*dx)) * (self.sigma(k,self.d[idys,idxs+1]) - self.sigma(k,self.d[idys,idxs-1]))
        elif direction == 'y':
            ray_depth_last = self.d.isel(y=xa.DataArray(idys-1,dims='z'),x=xa.DataArray(idxs,dims='z'))
            ray_depth_next = self.d.isel(y=xa.DataArray(idys+1,dims='z'),x=xa.DataArray(idxs,dims='z'))
            #dsigma = (1/(2*dx)) * (self.sigma(k,self.d[idys+1,idxs]) - self.sigma(k,self.d[idys-1,idxs]))

        dsigma = (1/(2*dx)) * (self.sigma(k,ray_depth_next) - self.sigma(k,ray_depth_last))

        return dsigma


    def wave(self,T,theta,d):
        """ Method computing generic wave properties

        Args:
            T (float): Wave period
            theta (float): radians. Wave direction

        Returns:
            k0 (float): wave number
            kx0 (float): wave number in x-direction
            ky0 (float): wave number in y-direction
        """
        g=self.g

        sigma = (2*np.pi)/T
        k0 = (sigma**2)/g

        # approximation of wave number according to Eckart (1952)
        alpha = k0*d
        k = (alpha/np.sqrt(np.tanh(alpha)))/d

        kx = k*np.cos(theta)
        ky = k*np.sin(theta)
        logger.info('wave: {}, {},{}'.format(k,kx,ky))
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

        """
        hvis iws:
            velg iws
            hvis iws invalid
               set left
        hvis ikke iws:
            sjekk ipx, ipy,
            whis ikke
                velg iws=left
        """

        valid_sides = ['left', 'right','top','bottom']

        if 'incoming_wave_side' in kwargs:
            i_w_side = kwargs['incoming_wave_side']

            if not i_w_side in valid_sides:
                logger.info('No initial position or side given. Left will be used.')
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
                #if 'ipx' or 'ipy' in kwargs:

        else:
            logger.info('No initial side given. Try with discrete points')
            try:
                ipx = kwargs['ipx']
                ipy = kwargs['ipy']

                # First check initial position x
                if type(ipx) is float:
                    ipx = np.ones(nb_wave_rays)*ipx
                    xs=ipx.copy()
                elif isinstance(ipx,np.ndarray):
                    assert nb_wave_rays == len(kwargs['ipx']), "Need same dimension on initial x-values"
                    xs=ipx.copy()
                else:
                    logger.error('ipx must be either float or numpy array. Terminating.')
                    sys.exit()

                if type(ipy) is float:
                    ipy = np.ones(nb_wave_rays)*ipy
                    ys=ipy.copy()
                elif isinstance(ipy,np.ndarray):
                    assert nb_wave_rays == len(kwargs['ipy']), "Need same dimension on initial y-values"
                    ys=ipy.copy()
                else:
                    logger.error('ipy must be either float or numpy array. Terminating.')
                    sys.exit()
            except:
                logger.info('No initial position ponts given. Left will be used')
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)


        self.xr[:,0] = xs
        self.yr[:,0] = ys


        #Theta0
        if type(theta0) is float or type(theta0) is int:
            theta0 = np.ones(nb_wave_rays)*theta0
        elif isinstance(theta0,np.ndarray):
            assert nb_wave_rays == len(theta0), "Initial values must have same dimension as number of wave rays"
        else:
            logger.error('Theta0 must be either float or numpy array. Terminating.')
            sys.exit()


        # set inital wave
        for i in range(nb_wave_rays):
            self.k[i,0], self.kx[i,0], self.ky[i,0] = self.wave(T=wave_period,
                                                                theta=theta0[i],
                                                                d=self.d.sel(y=ys[i],x=xs[i],method='nearest'))
        self.theta[:,0] = theta0


    def solve(self, solver=uts.RungeKutta4):
        """ Solve the geometrical optics equations numerically by means of the
            method of characteristics
        """

        if not callable(solver):
            raise TypeError('f is %s, not a solver' % type(solver))

        k = self.k
        kx= self.kx
        ky= self.ky
        xr= self.xr
        yr= self.yr
        theta= self.theta
        cg_i = self.cg_i

        dx = self.dx
        dy = self.dy

        U = self.U.data
        V = self.V.data

        #Compute velocity gradients
        logger.info('Assuming uniform horizontal resolution')
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
            idxs = np.array([self.find_nearest(x,xval) for xval in xr[:,n]])
            idys = np.array([self.find_nearest(y,yval) for yval in yr[:,n]])

            ray_depth = self.d.isel(y=xa.DataArray(idys,dims='z'),x=xa.DataArray(idxs,dims='z'))
            self.ray_depth[:,n] = ray_depth


            #======================================================
            ### numerical integration of the wave ray equations ###
            #======================================================

            # Compute group velocity
            cg_i[:,n+1] = self.c_intrinsic(k[:,n],d=ray_depth,group_velocity=True)

            # ADVECTION
            f_adv = uts.Advection(cg=cg_i[:,n+1], k=k[:,n], kx=kx[:,n], U=U[velocity_idt[n],idys,idxs])
            xr[:,n+1] = solver.advance(u=xr[:,n], f=f_adv,k=n,t=t)

            f_adv = uts.Advection(cg=cg_i[:,n+1], k=k[:,n], kx=ky[:,n], U=V[velocity_idt[n],idys,idxs])
            yr[:,n+1] = solver.advance(u=yr[:,n], f=f_adv, k=n, t=t)


            # EVOLUTION IN WAVE NUMBER
            self.dsigma_dx[:,n+1] = self.dsigma(k[:,n], idxs, idys, self.dx,direction='x')
            self.dsigma_dy[:,n+1] = self.dsigma(k[:,n], idxs, idys, self.dx,direction='y')

            f_wave_nb = uts.WaveNumberEvolution(d_sigma=self.dsigma_dx[:,n+1], kx=kx[:,n], ky=ky[:,n],
                                               dUkx=dudx[velocity_idt[n],idys,idxs], dUky=dvdx[velocity_idt[n],idys,idxs])
            kx[:,n+1] = solver.advance(u=kx[:,n], f=f_wave_nb,k=n, t=t)

            f_wave_nb = uts.WaveNumberEvolution(d_sigma=self.dsigma_dy[:,n+1], kx=kx[:,n], ky=ky[:,n],
                                               dUkx=dudy[velocity_idt[n],idys,idxs], dUky=dvdy[velocity_idt[n],idys,idxs])
            ky[:,n+1] = solver.advance(u=ky[:,n], f=f_wave_nb, k=n, t=t)

            # Compute k
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)

            # THETA
            theta[:,n+1] = np.arctan2(ky[:,n+1],kx[:,n+1])
            #keep angles between 0 and 2pi
            theta[:,n+1] = theta[:,n+1]%(2*np.pi)


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
                        ax3[0].plot(self.xr[id,:n+1],self.yr[id,:n+1],'-k')

                    ax3[0].plot(wt.xr[wr_id[2],idts],wt.yr[wr_id[2],idts],marker='s',ms=7,color='tab:red',linestyle='none')

                    ax3[0].xaxis.tick_top()

                    ax3[1].plot(-wt.ray_depth[wr_id[2],:1090], label=r'$d(x_r,y_r)$')
                    ax3[2].plot(wt.kx[wr_id[2],:1090], label=r'$k_x$')
                    ax3[2].plot(wt.ky[wr_id[2],:1090], label=r'$k_y$',c='tab:green')
                    ax3[3].plot(wt.theta[wr_id[2],:1090], label=r'$\theta$')

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
                    #fig3.savefig('/home/trygveh/documents/phd/papers/wave_ray_tracing/figures/POC.png',dpi=170)
                    #plt.show()


        self.dudy = dudy
        self.dudx = dudx
        self.dvdy = dvdy
        self.dvdx = dvdx
        self.k = k
        self.kx= kx
        self.ky= ky
        self.xr= xr
        self.yr= yr
        self.theta = theta
        self.cg_i = cg_i
        logging.info('Stoppet at time idt: {}'.format(velocity_idt[n]))

    def to_ds(self):
        with open('ray_metadata.json') as f:
            data = json.load(f)

        # relative time
        t = np.linspace(0,self.T,self.nt)
        ray_id = np.arange(self.nb_wave_rays)

        vars = make_xarray_dataArray(var=self.k, t=t,rays=ray_id,name='k',attribs=data['k'])
        print(vars)

    def to_NetCDF(self,fname,**kwargs):

        # relative time
        t = np.linspace(0,self.T,self.nt)


        if 'proj4' in kwargs:
            lons,lats = self.to_latlon(kwargs['proj4'])
        else:
            lons = np.zeros(self.nt)
            lats = lons.copy()

        with (Dataset(fname, 'w', format='NETCDF4')) as ncout:
            dim_time = ncout.createDimension('time',self.nt)
            dim_wave_ray = ncout.createDimension('ray_id',self.nb_wave_rays)

            nctime = ncout.createVariable('time','i4',('time',))

            # Set time value
            ##########################################################
            nctime.long_name = 'reference time for ray trajectory'
            nctime.units = 'seconds since start'
            nctime[:] = t

            varout = ncout.createVariable('k'    ,np.float32,('ray_id','time'))
            varout[:] = self.k
            varout = ncout.createVariable('kx'   ,np.float32,('ray_id','time'))
            varout[:] = self.kx
            varout = ncout.createVariable('ky'   ,np.float32,('ray_id','time'))
            varout[:] = self.ky
            varout = ncout.createVariable('xr'   ,np.float32,('ray_id','time'))
            varout[:] = self.xr
            varout = ncout.createVariable('yr'   ,np.float32,('ray_id','time'))
            varout[:] = self.yr
            varout = ncout.createVariable('theta',np.float32,('ray_id','time'))
            varout[:] = self.theta
            varout = ncout.createVariable('cg_i',np.float32,('ray_id','time'))
            varout[:] = self.cg_i
            varout = ncout.createVariable('lons',np.float32,('ray_id','time'))
            varout[:] = lons
            varout = ncout.createVariable('lats',np.float32,('ray_id','time'))
            varout[:] = lats

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
                    valid_x = (self.xr[i,:]>x0)*(self.xr[i,:]<xn)
                    if (np.any((self.yr[i,:][valid_x]>y0)*(self.yr[i,:][valid_x]<yn))):
                        hm[idy,idx]+=1

        if plot:
            plt.pcolormesh(xx,yy,hm)
            plt.colorbar()
            for i in range(0,self.nb_wave_rays):
                plt.plot(self.xr[i,:],self.yr[i,:],'-r',alpha=0.3)
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
            lons[i,:],lats[i,:] = pyproj.Transformer.from_proj(proj4,'epsg:4326', always_xy=True).transform(self.xr[i,:], self.yr[i,:])

        return lons, lats

if __name__ == '__main__':
    test = 'eddy' #lofoten, eddy, zero
    bathymetry = True

    if test=='lofoten':
        u_eastwards = xa.open_dataset('../current_forcing/u_eastwards.nc')
        v_northwards = xa.open_dataset('../current_forcing/v_northward.nc')
        U = u_eastwards.isel(time=1).to_array()[0].data
        V = v_northwards.isel(time=1).to_array()[0].data
        X = u_eastwards.X
        Y = u_eastwards.Y
        nx = U.shape[1]
        ny = U.shape[0]
        nb_wave_rays = 120
        dx=dy=800
        T = 3100 #Total duration
        print("T={}h".format(T/3600))
        nt = 300 # Nb time steps
        wave_period = 10

        X0, XN = X[0].data,X[-1].data
        Y0, YN = Y[0].data,Y[-1].data

        initial_position_x = float(0.5*(XN-X0))#np.arange(nb_wave_rays)
        initial_position_y = float(0.5*(YN-Y0))#np.arange(nb_wave_rays)

        d=None

    elif test=='eddy':
        idt0=15 #22
        ncin = xa.open_dataset('../notebooks/idealized_input.nc')
        U = ncin.U[idt0::,:,:]
        V = ncin.V[idt0::,:,:]
        X = ncin.x.data
        Y = ncin.y.data
        nx = len(Y)
        ny = len(X)
        dx=dy=X[1]-X[0]
        nb_wave_rays = 200#550#nx
        T = 3000
        print("T={}h".format(T/3600))
        nt = 300
        wave_period = 5
        X0, XN = X[0], X[-1]
        Y0, YN = Y[0], Y[-1]
        initial_position_x = float(0.5*(XN-X0))#np.arange(nb_wave_rays)
        initial_position_y = float(0.5*(YN-Y0))#np.arange(nb_wave_rays)

        if bathymetry:
            #d = ncin.bathymetry_bm.data
            d = ncin.bathymetry_1dy_slope.data


    elif test=='zero':
        idt0=15 #22
        ncin = xa.open_dataset('../notebooks/idealized_input.nc')
        U = ncin.U_zero[idt0::,:,:]
        V = ncin.V_zero[idt0::,:,:]
        X = ncin.x.data
        Y = ncin.y.data
        nx = len(Y)
        ny = len(X)
        dx=dy=X[1]-X[0]
        nb_wave_rays = 200#550#nx
        T = 1000
        print("T={}h".format(T/3600))
        nt = 300
        wave_period = 20
        X0, XN = X[0], X[-1]
        Y0, YN = Y[0], Y[-1]

        initial_position_x = float(0.5*(XN-X0))#np.arange(nb_wave_rays)
        initial_position_y = float(0.5*(YN-Y0))#np.arange(nb_wave_rays)

        if bathymetry:
            d = ncin.bathymetry_bm.data
            #d = ncin.bathymetry_1dy_slope.data

    i_w_side = 'left'#'top'
    if i_w_side == 'left':
        theta0 = 0.12 #Initial wave propagation direction
        theta0 = np.linspace(0,np.pi,nb_wave_rays) #Initial wave propagation direction
    elif i_w_side == 'top':
        theta0 = 1.5*np.pi#0#np.pi/8 #Initial wave propagation direction
    elif i_w_side == 'right':
        theta0 = 1*np.pi#0#np.pi/8 #Initial wave propagation direction
    elif i_w_side == 'bottom':
        theta0 = 0.5*np.pi#0#np.pi/8 #Initial wave propagation direction


    if bathymetry:

        wt = Wave_tracing(U, V, nx, ny, nt,T,dx,dy, nb_wave_rays=nb_wave_rays,
                            domain_X0=X0, domain_XN=XN,
                            domain_Y0=Y0, domain_YN=YN,
                            d=d,DEBUG=False,)
    else:
        wt = Wave_tracing(U, V, nx, ny, nt,T,dx,dy, nb_wave_rays=nb_wave_rays,
                            domain_X0=X0, domain_XN=XN,
                            domain_Y0=Y0, domain_YN=YN,
                            DEBUG=True)


    wt.set_initial_condition(wave_period=wave_period, theta0=theta0,
                             incoming_wave_side=i_w_side,
                             ipx=initial_position_x,ipy=initial_position_y)
    #wt.solve(solver=uts.ForwardEuler)
    wt.solve(solver=uts.RungeKutta4)



    ### PLOTTING ###
    fig,ax = plt.subplots(figsize=(16,6))
    if test=='lofoten':
        vorticity = wt.dvdx-wt.dudy
        pc=ax.pcolormesh(wt.x,wt.y,vorticity[0,:,:],shading='auto',cmap='bwr',
                         vmin=-0.0003,vmax=0.0003)
    elif test=='eddy':
        vorticity = wt.dvdx-wt.dudy
        pc=ax.pcolormesh(wt.x,wt.y,vorticity[0,:,:],shading='auto',cmap='bwr',
                         vmin=-0.0004,vmax=0.0004)
    elif test=='zero' and bathymetry:
        pc=ax.contourf(wt.x,wt.y,-d,shading='auto',cmap='viridis',levels=25)

    ax.plot(wt.xr[:,0],wt.yr[:,0],'o')
    step=2
    for i in range(0,wt.nb_wave_rays,step):
        ax.plot(wt.xr[i,:],wt.yr[i,:],'-k')

    idts = np.arange(0,nt,40)
    #ax.plot(wt.xr[:,idts],wt.yr[:,idts],'--k')
    cb = fig.colorbar(pc)


    if test == 'lofoten':
        # Georeference
        proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70' #NK800
        lons,lats=wt.to_latlon(proj4)

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig2, ax2 = plt.subplots(frameon=False,figsize=(7,7),subplot_kw={'projection': ccrs.Mercator()})

        pc2=ax2.pcolormesh(u_eastwards.lon,u_eastwards.lat,vorticity[0,:,:],shading='auto',cmap='bwr',
                         vmin=-0.0003,vmax=0.0003, transform=ccrs.PlateCarree())

        for i in range(wt.nb_wave_rays):
            ax2.plot(lons[i,:],lats[i,:],'-k',transform=ccrs.PlateCarree(),alpha=0.6)

        ax2.coastlines()
        cb2 = fig2.colorbar(pc2, extend='both')

    plot_single_ray = True
    if plot_single_ray:
        ray_id = 105
        idts = np.arange(100,300,70)
        fig3,ax3 = plt.subplots(nrows=4,ncols=1,figsize=(16,10),gridspec_kw={'height_ratios': [3, 1,1,1]})

        pc=ax3[0].contourf(wt.x,wt.y,-wt.d,shading='auto',cmap=cm.deep,levels=25)
        ax3[0].plot(wt.xr[ray_id,:],wt.yr[ray_id,:],'-k')
        ax3[0].plot(wt.xr[ray_id,0],wt.yr[ray_id,0],'o')
        ax3[0].plot(wt.xr[ray_id,idts],wt.yr[ray_id,idts],'rs')

        ax3[1].plot(wt.ray_depth[ray_id,:],label='depth');
        ax3[2].plot(wt.kx[ray_id,:],label='kx')
        ax3[3].plot(wt.ky[ray_id,:], label='ky')

        ax3[2].sharex(ax3[1])
        ax3[3].sharex(ax3[1])

        cb3 = fig3.colorbar(pc,ax=ax3[0])

        for aax in [ax3[1],ax3[2],ax3[3]]:
            for idt in idts:
                aax.axvline(idt,c='r')
            aax.grid()

    plt.show()
