import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xa
import pyproj
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(filename='wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')

# TODO:
#   - fix direction if >45 degrees. What is the nearest grid point in dm and ds


class Wave_tracing_FE():
    """ Class for tracing wave rays according to the geometrical optics
    approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy, wave_period, theta0,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 incoming_wave_side,temporal_evolution=False, T0=None,
                 d=None):
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
            wave_period (float): Wave period.
            theta0 (rad): Wave initial direction. In radians.
                         (0,.5*pi,pi,1.5*pi) correspond to going
                         (right, up, left, down).
            nb_wave_rays (int): Number of wave rays to track. NOTE: Should be
                                equal or less to either nx or ny.
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction
            incoming_wave_side (str): side for incoming wave direction
                                        [left, right, top, bottom]

            temporal_evolution (bool): flag if velocity field should change in time
        """
        self.g = 9.81
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.wave_period = wave_period
        self.theta0 = theta0
        self.nb_wave_rays = nb_wave_rays

        self.domain_X0 = domain_X0 # left side
        self.domain_XN = domain_XN # right side
        self.domain_Y0 = domain_Y0 # bottom
        self.domain_YN = domain_YN # top
        self.i_w_side = incoming_wave_side

        self.temporal_evolution = temporal_evolution

        if d is not None:
            self.d = self.check_bathymetry(d)
            self.ray_depth = np.zeros((nb_wave_rays,nt))
            self.dd_ds= np.zeros((nb_wave_rays,nt))
        else:
            logging.warning('Hardcoding bathymetry if not given. Should be fixed')
            self.d = np.ones((ny,nx))*30000
            self.ray_depth = np.zeros((nb_wave_rays,nt))
            self.dd_ds= np.zeros((nb_wave_rays,nt))

        # Setting up X and Y domain
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        # Setting up the wave rays
        self.xr = np.zeros((nb_wave_rays,nt))
        self.yr = np.zeros((nb_wave_rays,nt))
        self.kx = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.ky = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.k = np.zeros((nb_wave_rays,nt))#np.zeros(nt)
        self.theta = np.ma.zeros((nb_wave_rays,nt))
        self.dudm = np.ma.zeros((nb_wave_rays,nt))
        self.dvdm = np.ma.zeros((nb_wave_rays,nt))

        # Change in intrinsic frequency due to depth
        self.dsigma_dm = np.ma.zeros((nb_wave_rays,nt))
        self.dsigma_ds = np.ma.zeros((nb_wave_rays,nt))



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
            logging.info('lengtsh: {}, {}'.format(len(self.velocity_idt),nt))

    def check_bathymetry(self,d):

        if np.any(d < 0) and np.any(d > 0):
            logger.warning('Depth is defined as positive. Thus, negative depth will be treated as Land.')
            d[d<0] = 0
            return d

        if np.any(d < 0):
            logger.warning('Depth is defined as positive. Hence taking absolute value of input.')
            d = np.abs(d)

        d[d==0] = np.nan
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
            print('Here {}'.format(c_in))
            n=0.5
        else:
            #logger.info(d)
            c_in = np.sqrt((g/k)*np.tanh(k*d)) #intrinsic
            n = 0.5 * (1 + (2*k*d)/np.sinh(2*k*d))

        if group_velocity:
            return c_in*n
            #return 0.5*np.sqrt(g/k)
        else:
            return c_in
            #return np.sqrt(g/k)



    def wave(self,T,theta):
        """ Method computing deep water wave properties

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

        kx0 = k0*np.cos(theta)
        ky0 = k0*np.sin(theta)
        logger.info('wave: {}, {},{}'.format(k0,kx0,ky0))
        return k0,kx0,ky0


    def set_initial_condition(self):
        """ Setting inital conditions before solving numerically.
        """
        k0, kx0, ky0 = self.wave(self.wave_period, self.theta0)

        if self.i_w_side == 'left':
            self.xr[:,0]=self.domain_X0
            self.yr[:,0]=np.linspace(self.domain_Y0, self.domain_YN, self.nb_wave_rays)
        elif self.i_w_side == 'right':
            self.xr[:,0]=self.domain_XN
            self.yr[:,0]=np.linspace(self.domain_Y0, self.domain_YN, self.nb_wave_rays)
        elif self.i_w_side == 'top':
            self.xr[:,0]=np.linspace(self.domain_X0, self.domain_XN, self.nb_wave_rays)
            self.yr[:,0]=self.domain_YN
        elif self.i_w_side == 'bottom':
            self.xr[:,0]=np.linspace(self.domain_X0, self.domain_XN, self.nb_wave_rays)
            self.yr[:,0]=self.domain_Y0
        else:
            logger.error('Invalid initial wave direcion. Terminating.')
            sys.exit()

        self.kx[:,0]=kx0
        self.ky[:,0]=ky0
        self.k[:,0]=k0
        logger.info('theta: {}, {}'.format(self.theta0,np.arctan(ky0/kx0)))
        self.theta[:,0] = self.theta0#np.arctan(ky0/kx0) # self.theta0

    def find_idx_idy_relative_to_wave_direction(self, idxs, idys, theta, orthogonal=False):
        if orthogonal:
            phi = (theta - (0.5*np.pi))%(2*np.pi)
        else:
            phi = theta

        idxs_p1 = idxs.copy()
        idys_p1 = idys.copy()

        idxs_p1[np.where(~np.logical_and(phi>=(np.pi/4), phi<=((7*np.pi)/4)))] += 1 #Q1
        idys_p1[np.where(np.logical_and(phi>((1*np.pi)/4), phi<((3*np.pi)/4)))] += 1 #Q2
        idxs_p1[np.where(np.logical_and(phi>((3*np.pi)/4), phi<((5*np.pi)/4)))] -= 1 #Q3
        idys_p1[np.where(np.logical_and(phi>((5*np.pi)/4), phi<((7*np.pi)/4)))] -= 1 #Q4

        # Fixing indices outside domain
        idxs_p1[idxs_p1<0] = 0
        idxs_p1[idxs_p1>=self.nx] = self.nx-1
        idys_p1[idys_p1<0] = 0
        idys_p1[idys_p1>=self.ny] = self.ny-1

        return(idxs_p1, idys_p1)


    def solve(self):
        """ Solve the geometrical optics equations numerically
        """

        k = self.k
        kx= self.kx
        ky= self.ky
        xr= self.xr
        yr= self.yr
        theta= self.theta
        logging.info(self.U.dims)

        dx = self.dx
        dy = self.dy

        U = self.U.data
        V = self.V.data

        #Compute velocity gradients

        logger.warning('Assuming uniform horizontal resolution')
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


        x= self.x
        y= self.y
        dt= self.dt
        nt = self.nt
        velocity_idt = self.velocity_idt


        dt2 = dt/2.0
        counter=0
        for n in range(0,nt-1):

            # find indices for each wave ray
            idxs = np.array([self.find_nearest(x,xval) for xval in xr[:,n]])
            idys = np.array([self.find_nearest(y,yval) for yval in yr[:,n]])
            self.ray_depth[:,n] = self.d[idys,idxs]

            # compute the change in direction
            if self.i_w_side == 'right' or self.i_w_side == 'left':

                # in m direction (orthogonal to wave propagation direction)
                idxs_dm_p1, idys_dm_p1 = self.find_idx_idy_relative_to_wave_direction(idxs=idxs,
                                         idys=idys, theta=theta[:,n],orthogonal=True)
                dm=self.dx


                dudm = (U[velocity_idt[n],idys_dm_p1,idxs_dm_p1] - U[velocity_idt[n],idys,idxs])/dm
                dvdm = (V[velocity_idt[n],idys_dm_p1,idxs_dm_p1] - V[velocity_idt[n],idys,idxs])/dm

                self.dudm[:,n+1]=dudm
                self.dvdm[:,n+1]=dvdm

                dsigma_dm = (-1)*(self.c_intrinsic(k[:,n],d=self.d[idys_dm_p1,idxs_dm_p1]) -
                                self.c_intrinsic(k[:,n],d=self.d[idys,idxs]) )/dm
                self.dsigma_dm[:,n+1] = dsigma_dm

                # in s direction (wave propagation direction)
                idxs_ds_p1, idys_ds_p1 = self.find_idx_idy_relative_to_wave_direction(idxs=idxs,
                                         idys=idys, theta=theta[:,n],orthogonal=False)
                ds=self.dx

                dd_ds = (-1)*(self.d[idys_ds_p1,idxs_ds_p1] - self.d[idys,idxs])/ds

                dsigma_ds = (-1)*(self.c_intrinsic(k[:,n],d=self.d[idys_ds_p1,idxs_ds_p1]) -
                                self.c_intrinsic(k[:,n],d=self.d[idys,idxs]) )/ds

                self.dsigma_ds[:,n+1] = dsigma_ds
                self.dd_ds[:,n+1] = dd_ds


                #logging.info(idys_ds_p1)
                #logging.info(idys_dm_p1)
                #logging.info(self.c_intrinsic(k[:,n],d=self.d[idys_dm_p1,idxs_dm_p1]))
                #logging.info(self.d[idys_dm_p1,idxs_dm_p1])
                #break


                """
                #dm
                id_p1 = idys +1
                id_p1[id_p1>=self.ny] = self.ny-1
                dm = y[id_p1] - y[idys]

                #ds

                #logging.info('TEST: {}'.format(dm))

                dudm = (U[velocity_idt[n],id_p1,idxs] - U[velocity_idt[n],idys,idxs])/dm
                dvdm = (V[velocity_idt[n],id_p1,idxs] - V[velocity_idt[n],idys,idxs])/dm

                self.dudm[:,n+1]=dudm
                self.dvdm[:,n+1]=dvdm

                dsigma_dm = ( self.c_intrinsic(k[:,n],d=self.d[id_p1,idxs]) -
                                self.c_intrinsic(k[:,n],d=self.d[idys,idxs]) )/dm
                self.dsigma_dm[:,n+1] = dsigma_dm
                """

            elif self.i_w_side == 'top' or self.i_w_side == 'bottom':
                #dm
                id_p1 = idxs +1
                id_p1[id_p1>=self.nx] = self.nx-1
                dm = x[id_p1] - x[idxs]



                dudm = (U[velocity_idt[n],idys,id_p1] - U[velocity_idt[n],idys,idxs])/dm
                dvdm = (V[velocity_idt[n],idys,id_p1] - V[velocity_idt[n],idys,idxs])/dm
                self.dudm[:,n+1]=dudm
                self.dvdm[:,n+1]=dvdm


                dsigma_dm = (self.c_intrinsic(k[:,n],d=self.d[idys,id_p1]) -
                                self.c_intrinsic(k[:,n],d=self.d[idys,idxs]) )/dm
                self.dsigma_dm[:,n+1] = dsigma_dm

            theta[:,n+1] = theta[:,n] - dt*(1/k[:,n])*(dsigma_dm + kx[:,n]*dudm + ky[:,n]*dvdm)#  varying depth
            theta[:,n+1] = theta[:,n+1]%(2*np.pi) #keep angles between 0 and 2pi

            #theta[:,n+1] = theta[:,n] - dt*(1/k[:,n])*(kx[:,n]*dudm + ky[:,n]*dvdm)#  deep water
            #theta[:,n+1] = np.arctan(ky[:,n]/kx[:,n])

            # Compute group velocity
            cg_i = self.c_intrinsic(k[:,n],d=self.d[idys,idxs],group_velocity=True)
            #cg_i = self.c_intrinsic(k[:,n],group_velocity=True) # deep water

            cg_i_x =  cg_i*np.cos(theta[:,n])
            cg_i_y =  cg_i*np.sin(theta[:,n])

            # Advection
            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[velocity_idt[n],idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[velocity_idt[n],idys,idxs])

            # Evolution in wave number
            kx[:,n+1] = kx[:,n] - dt*(dsigma_ds*dd_ds + kx[:,n]*dudx[velocity_idt[n],idys,idxs] + ky[:,n]*dvdx[velocity_idt[n],idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(dsigma_ds*dd_ds + kx[:,n]*dudy[velocity_idt[n],idys,idxs] + ky[:,n]*dvdy[velocity_idt[n],idys,idxs])
            #kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[velocity_idt[n],idys,idxs] + ky[:,n]*dvdx[velocity_idt[n],idys,idxs])
            #ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[velocity_idt[n],idys,idxs] + ky[:,n]*dvdy[velocity_idt[n],idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)

            # Logging purposes
            counter += 1
            if counter in range(850,930,50):
                wr_id = 75
                #logging.info(np.any(np.isnan(U[idys,idxs])))
                #logging.info(dt*n)
                #dm = np.sqrt(np.gradient(xr[:,n])**2 + np.gradient(yr[:,n])**2)
                #logging.info('dm:{}'.format(len(dm)))
                #logging.info('V: {}'.format(V[velocity_idt[n],idys,idxs]))
                #logging.info(np.gradient(U[velocity_idt[n],idys,idxs],dm))
                #logging.info([idd for idd in idxs])

                #logging.info(cg_i[wr_id])
                logging.info("n: {}, dsigma_dm:{}".format(n+1,dsigma_dm[wr_id]))
                logging.info("dsigma_ds dd_ds: {}".format(dsigma_ds[wr_id]*dd_ds[wr_id]))
                logging.info("dd_ds: {}".format(dd_ds[wr_id]))
                #logging.info("theta:{}".format(theta[:,n+1]))
                #logging.info("x:{}".format(x))
                #break
                fig,ax = plt.subplots(figsize=(16,6))
                pc=ax.pcolormesh(self.x,self.y,-self.d,shading='auto',cmap='viridis')
                #step=2
                #for i in range(0,wt.nb_wave_rays,step):
                #    ax.plot(self.xr[i,:n+1],self.yr[i,:n+1],'-k')

                ax.plot(self.xr[wr_id,:n+1],self.yr[wr_id,:n+1],'-k')

                ax.plot(self.x[idxs[wr_id]],self.y[idys[wr_id]],'bo')
                ax.plot(self.x[idxs_dm_p1[wr_id]],self.y[idys_dm_p1[wr_id]],'rs',alpha=0.4)
                ax.plot(self.x[idxs_ds_p1[wr_id]],self.y[idys_ds_p1[wr_id]],'ms',alpha=0.4)
                cb = fig.colorbar(pc)
                ax.set_title('idt: {}, theta: {}, cg:{}'.format(counter,theta[wr_id,n+1],cg_i[wr_id]))
                phi = (theta[wr_id,n+1] - (0.5*np.pi))%(2*np.pi)
                print('theta: {}'.format(theta[wr_id,n+1],phi))
                print('depth: {}'.format(self.d[idys[wr_id],idxs[wr_id]]))

                plt.show()


            """ FE
            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[idys,idxs])


            kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[idys,idxs] + ky[:,n]*dvdx[idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[idys,idxs] + ky[:,n]*dvdy[idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)
            """
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
        logging.info('Stoppet at time idt: {}'.format(velocity_idt[n]))

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
                        #print(idx,idy,'OK')
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
    import xarray as xa


    """
    X = 3000 #m
    Y = 1500 #m
    nx, ny = (200,101)
    x = np.linspace(0, X, nx)
    y = np.linspace(-Y, Y, ny)
    dx,dy = 15,15

    U0 = -.1
    Y = np.max(y)*2
    U_y = U0*np.cos((np.pi*y)/Y)**2

    U = (np.ones((nx,ny))*U_y).T
    V = np.zeros((ny,nx))
    """

    test = 'eddy' #lofoten, eddy, zero
    bathymetry = True

    if test=='lofoten':
        u_eastwards = xa.open_dataset('u_eastwards.nc')
        v_northwards = xa.open_dataset('v_northward.nc')
        U = u_eastwards.isel(time=1).to_array()[0].data
        V = v_northwards.isel(time=1).to_array()[0].data
        X = u_eastwards.X
        Y = u_eastwards.Y
        nx = U.shape[1]
        ny = U.shape[0]
        nb_wave_rays = 120
        dx=dy=800
        T = 31000 #Total duration
        print("T={}h".format(T/3600))
        nt = 3000 # Nb time steps
        wave_period = 10



        X0, XN = X[0].data,X[-1].data
        Y0, YN = Y[0].data,Y[-1].data

    elif test=='eddy':
        idt0=15 #22
        ncin = xa.open_dataset('idealized_input.nc')
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
        nt = 1900
        wave_period = 7
        #X0, XN = Y[0], Y[-1] #NOTE THIS
        #Y0, YN = X[0], X[-1] #NOTE THIS
        X0, XN = Y[0], Y[-1] #NOTE THIS
        Y0, YN = X[0], X[-1] #NOTE THIS

        if bathymetry:
            d = ncin.bathymetry_1dy_slope.data

    elif test=='zero':
        idt0=15 #22
        ncin = xa.open_dataset('idealized_input.nc')
        U = ncin.U_zero[idt0::,:,:]
        V = ncin.V_zero[idt0::,:,:]
        X = ncin.x.data
        Y = ncin.y.data
        nx = len(Y)
        ny = len(X)
        dx=dy=X[1]-X[0]
        nb_wave_rays = 200#550#nx
        T = 2700
        print("T={}h".format(T/3600))
        nt = 1700
        wave_period = 10
        #X0, XN = Y[0], Y[-1] #NOTE THIS
        #Y0, YN = X[0], X[-1] #NOTE THIS
        X0, XN = Y[0], Y[-1] #NOTE THIS
        Y0, YN = X[0], X[-1] #NOTE THIS

        if bathymetry:
            #d = ncin.bathymetry_bm.data
            d = ncin.bathymetry_1dy_slope.data

    i_w_side = 'left'#'top'
    if i_w_side == 'left':
        theta0 = 0 #Initial wave propagation direction
    elif i_w_side == 'top':
        theta0 = 1.5*np.pi#0#np.pi/8 #Initial wave propagation direction
    elif i_w_side == 'right':
        theta0 = 1*np.pi#0#np.pi/8 #Initial wave propagation direction
    elif i_w_side == 'bottom':
        theta0 = 0.5*np.pi#0#np.pi/8 #Initial wave propagation direction


    if bathymetry:
        wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, theta0, nb_wave_rays=nb_wave_rays,
                            domain_X0=X0, domain_XN=XN,
                            domain_Y0=Y0, domain_YN=YN,
                            incoming_wave_side=i_w_side,d=d)
    else:
        wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, theta0, nb_wave_rays=nb_wave_rays,
                            domain_X0=X0, domain_XN=XN,
                            domain_Y0=Y0, domain_YN=YN,
                            incoming_wave_side=i_w_side)
    wt.set_initial_condition()
    wt.solve()



    ### PLOTTING ###
    fig,ax = plt.subplots(figsize=(16,6))
    if test=='lofoten':
        pc=ax.pcolormesh(X,Y,wt.U.isel(time=0),shading='auto')
    elif test=='eddy':
        vorticity = wt.dvdx-wt.dudy
        pc=ax.pcolormesh(Y,X,vorticity[0,:,:],shading='auto',cmap='bwr',
                         vmin=-0.0004,vmax=0.0004)
    elif test=='zero' and bathymetry:
        pc=ax.contourf(Y,X,-d,shading='auto',cmap='viridis',levels=25)
        #pc=ax.contourf(Y,X,np.gradient(-d)[1],shading='auto',cmap='viridis')

    ax.plot(wt.xr[:,0],wt.yr[:,0],'o')
    step=2
    for i in range(0,wt.nb_wave_rays,step):
        ax.plot(wt.xr[i,:],wt.yr[i,:],'-k')
    cb = fig.colorbar(pc)
                #fig.savefig('T3')

    if test == 'lofoten':
        # Georeference
        proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70' #NK800
        lons,lats=wt.to_latlon(proj4)

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig2, ax2 = plt.subplots(frameon=False,figsize=(7,7),subplot_kw={'projection': ccrs.Mercator()})

        for i in range(wt.nb_wave_rays):
            ax2.plot(lons[i,:],lats[i,:],'-k',transform=ccrs.PlateCarree())

        ax2.coastlines()

    plt.show()
