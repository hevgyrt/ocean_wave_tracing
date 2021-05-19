import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')

class Wave_tracing_FE():
    """ Class for tracing wave rays according to the geometrical optics
    approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy, wave_period, theta0,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN):
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
            theta0 (rad): Wave initial direction. In radians
            nb_wave_rays (int): Number of wave rays to track
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction
        """
        self.g = 9.81
        self.U = U
        self.V = V
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.wave_period = wave_period
        self.theta0 = theta0
        self.nb_wave_rays = nb_wave_rays

        self.domain_X0 = domain_X0
        self.domain_XN = domain_XN
        self.domain_Y0 = domain_Y0
        self.domain_YN = domain_YN

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

        # Time
        self.dt = T/nt
        self.t = np.linspace(0,T,nt)

        # Compute velocity gradients
        self.dudy, self.dudx = np.gradient(U,dx)
        self.dvdy, self.dvdx = np.gradient(V,dy)


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

    def c_intrinsic(self,k,group_velocity=False):
        """ Method computing intrinsic wave phase and group velocity according
        to deep water dispersion relation

        Args:
            k: wave number
            group_velocity (bool): returns group velocity (True) or phase
                velocity (False)

        Returns:
            instrinsic velocity (float): group or phase velocity depending on
                flag
        """
        g=self.g
        if group_velocity:
            return 0.5*np.sqrt(g/k)
        else:
            return np.sqrt(g/k)

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
        return k0,kx0,ky0


    def set_initial_condition(self):
        """ Setting inital conditions before solving numertically.
        """
        k0, kx0, ky0 = self.wave(self.wave_period, self.theta0)

        self.xr[:,0]=self.domain_X0
        self.yr[:,0]=np.linspace(self.domain_Y0, self.domain_YN, self.nb_wave_rays)
        self.kx[:,0]=kx0
        self.ky[:,0]=ky0
        self.k[:,0]=k0
        self.theta[:,0] = np.arctan(ky0/kx0) # self.theta0

#    def fx_k(self, k,t):
#        return arg1 + arg2

    def solve(self):
        """ Solve the geometrical optics equations numerically
        """
        k = self.k
        kx= self.kx
        ky= self.ky
        xr= self.xr
        yr= self.yr
        theta= self.theta
        U= self.U
        V= self.V
        dudx= self.dudx
        dudy= self.dudy
        dvdx = self.dvdx
        dvdy = self.dvdy
        x= self.x
        y= self.y
        dt= self.dt
        nt = self.nt

        dt2 = dt/2.0
        counter=0
        for n in range(0,nt-1):

            #theta[:,n+1] = theta[:,n] - dt*dudm
            theta[:,n+1] = np.arctan(ky[:,n]/kx[:,n])

            cg_i = self.c_intrinsic(k[:,n],group_velocity=True)

            cg_i_x =  cg_i*np.cos(theta[:,n])
            cg_i_y =  cg_i*np.sin(theta[:,n])

            idxs = np.array([self.find_nearest(x,xval) for xval in xr[:,n]])
            idys = np.array([self.find_nearest(y,yval) for yval in yr[:,n]])

#            K1_xr = dt*self.f(xr[:,n], t[k])
#            K2_xr = dt*self.f(xr[:,n] + 0.5*K1, t[k] + dt2)
#            K3_xr = dt*self.f(xr[:,n] + 0.5*K2, t[k] + dt2)
#            K4_xr = dt*self.f(xr[:,n] + K3, t[k] + dt)
#            u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
            #logging.info("xr:{}".format(xr[:,n].shape))
            #logging.info("yr:{}".format(yr.shape))
            #logging.info("idxs:{}".format(idxs.shape))
            #logging.info("idys:{}".format(idys.shape))
            #logging.info("cg_x:{}".format(cg_i_x.shape))
            #logging.info("U:{}, nbx: {}, nby: {}".format(U.shape,idxs.shape,idys.shape))
            #logging.info("Uidx:{}".format(U[idys,idxs].shape))
            #fig,ax = plt.subplots(figsize=(16,6))
            #pc=ax.pcolormesh(x,y,U);plt.show()


            #print(idxs)
            #print(idys)
            #print(xr.shape, U.shape, cg_i_x.shape)
            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[idys,idxs])


            kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[idys,idxs] + ky[:,n]*dvdx[idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[idys,idxs] + ky[:,n]*dvdy[idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)
            counter += 1
            if counter==1:
                logging.info(np.any(np.isnan(U[idys,idxs])))
                #logging.info([idd for idd in idys])
                #logging.info([idd for idd in idxs])
                #logging.info(idys)
                #logging.info("idxs:{}".format(idys))
                #logging.info("x:{}".format(x))


            """ FE
            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[idys,idxs])


            kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[idys,idxs] + ky[:,n]*dvdx[idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[idys,idxs] + ky[:,n]*dvdy[idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)
            """

        self.k = k
        self.kx= kx
        self.ky= ky
        self.xr= xr
        self.yr= yr
        self.theta = theta

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
    u_eastwards = xa.open_dataset('u_eastwards.nc')
    v_northwards = xa.open_dataset('v_northward.nc')
    U = u_eastwards.isel(time=1).to_array()[0].data
    V = v_northwards.isel(time=1).to_array()[0].data

    X = u_eastwards.X
    Y = u_eastwards.Y
    nx = U.shape[1]
    ny = U.shape[0]
    nb_wave_rays = 120
    #nx = 100
    #ny=40
    dx=dy=800

    T = 21000
    print("T={}h".format(T/3600))
    nt = 2000
    wave_period = 10
    theta0 = 0#np.pi/8
    wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, theta0, nb_wave_rays=nb_wave_rays,
                        domain_X0=X[0].data, domain_XN=X[-1].data,
                        domain_Y0=Y[0].data, domain_YN=Y[-1].data)
    wt.set_initial_condition()
    wt.solve()
    fig,ax = plt.subplots(figsize=(16,6))
    #pc=ax.pcolormesh(wt.x,wt.y,wt.U)
    #pc=ax.pcolormesh(np.arange(len(u_eastwards.X))*800,np.arange(len(u_eastwards.Y))*800,wt.U)
    pc=ax.pcolormesh(X,Y,wt.U)
    for i in range(wt.nb_wave_rays):
        #for i in range(wt.ny):
        #ax.plot(X[0].data+wt.xr[i,:],wt.yr[i,:],'-k')
        ax.plot(wt.xr[i,:],wt.yr[i,:],'-k')
    cb = fig.colorbar(pc)
                #fig.savefig('T3')
    plt.show()
