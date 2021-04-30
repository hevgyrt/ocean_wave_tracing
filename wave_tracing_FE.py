import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')

class Wave_tracing_FE():
    def __init__(self, U, V, nx, ny, nt,T,dx,dy, wave_period, theta0,
                 domain_X0, domain_XN, domain_Y0, domain_YN):
        """
        U (2d) = eastward velocity

        nx: number of grid points x-dir
        ny: number of grid points y-dir
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

        self.domain_X0 = domain_X0
        self.domain_XN = domain_XN
        self.domain_Y0 = domain_Y0
        self.domain_YN = domain_YN


        self.x = np.linspace(domain_X0, domain_XN, nx)
        #self.y = np.linspace(0, ny*dy, ny)
        self.y = np.linspace(domain_Y0, domain_YN, ny)
        self.xr = np.zeros((ny,nt))
        self.yr = np.zeros((ny,nt))
        self.dt = T/nt
        self.t = np.linspace(0,T,nt)
        self.kx = np.zeros((ny,nt))#np.zeros(nt)
        self.ky = np.zeros((ny,nt))#np.zeros(nt)
        self.k = np.zeros((ny,nt))#np.zeros(nt)
        self.theta = np.ma.zeros((ny,nt))



        self.dudy, self.dudx = np.gradient(U,dx)
        self.dvdy, self.dvdx = np.gradient(V,dy)

    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def c_intrinsic(self,k,group_velocity=False):
        g=self.g
        if group_velocity:
            return 0.5*np.sqrt(g/k)
        else:
            return np.sqrt(g/k)

    def wave(self,T,theta):
        """ DEEP WATER """
        g=self.g

        sigma = (2*np.pi)/T
        k0 = (sigma**2)/g

        kx0 = k0*np.cos(theta)
        ky0 = k0*np.sin(theta)
        return k0,kx0,ky0


    def set_initial_condition(self):
        k0, kx0, ky0 = self.wave(self.wave_period, self.theta0)

        self.xr[:,0]=0
        self.yr[:,0]=self.y
        self.kx[:,0]=kx0
        self.ky[:,0]=ky0
        self.k[:,0]=k0
        self.theta[:,0] = np.arctan(ky0/kx0) # self.theta0

#    def fx_k(self, k,t):
#        return arg1 + arg2

    def solve(self):
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
            #print(U)
            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[idys,idxs])


            kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[idys,idxs] + ky[:,n]*dvdx[idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[idys,idxs] + ky[:,n]*dvdy[idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)
            counter += 1
            #if counter==5:
            #    break

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


#if __name__ == '__main__()':
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
#nx = 100
#ny=40
dx=dy=800

T = 40000
print("T={}h".format(T/3600))
nt = 2500
wave_period = 10
theta0 = 0#np.pi/8
wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, theta0,
                    domain_X0=X[0].data, domain_XN=X[-1].data,
                    domain_Y0=Y[0].data, domain_YN=Y[-1].data)
wt.set_initial_condition()
wt.solve()

fig,ax = plt.subplots(figsize=(16,6))
#pc=ax.pcolormesh(wt.x,wt.y,wt.U)
#pc=ax.pcolormesh(np.arange(len(u_eastwards.X))*800,np.arange(len(u_eastwards.Y))*800,wt.U)
pc=ax.pcolormesh(X,Y,wt.U)
for i in range(5,wt.ny-1,5):
    ax.plot(X[0].data+wt.xr[i,:],wt.yr[i,:],'-k')

ax.set_xlim([X[0].data,X[-1].data])
ax.set_ylim([Y[0].data,Y[-1].data])
idts = np.arange(0,wt.nt,40)
#ax.plot(wt.xr[:,idts],wt.yr[:,idts],'--k')
cb = fig.colorbar(pc)
#fig.savefig('T3')
plt.show()
