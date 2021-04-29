import numpy as np
import matplotlib.pyplot as plt

class Wave_tracing_FE():
    def __init__(self, U, V, nx, ny, nt,T,dx,dy, wave_period, theta0):
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

        self.x = np.linspace(0, nx*dx, nx)
        self.y = np.linspace(0, ny*dy, ny)
        self.xr = np.zeros((ny,nt))
        self.yr = np.zeros((ny,nt))
        self.dt = T/nt
        self.t = np.linspace(0,T,nt)
        self.kx = np.zeros((ny,nt))#np.zeros(nt)
        self.ky = np.zeros((ny,nt))#np.zeros(nt)
        self.k = np.zeros((ny,nt))#np.zeros(nt)
        self.theta = np.ma.zeros((ny,nt))



        self.dudy, self.dudx = np.gradient(U)
        self.dvdy, self.dvdx = np.gradient(V)

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

        for n in range(0,nt-1):

            #theta[:,n+1] = theta[:,n] - dt*dudm
            theta[:,n+1] = np.arctan(ky[:,n]/kx[:,n])

            cg_i = self.c_intrinsic(k[:,n],group_velocity=True)

            cg_i_x =  cg_i*np.cos(theta[:,n])
            cg_i_y =  cg_i*np.sin(theta[:,n])

            idxs = np.array([self.find_nearest(x,xval) for xval in xr[:,n]])
            idys = np.array([self.find_nearest(y,yval) for yval in yr[:,n]])

            xr[:,n+1] = xr[:,n] + dt*(cg_i_x+U[idys,idxs])
            yr[:,n+1] = yr[:,n] + dt*(cg_i_y+V[idys,idxs])


            kx[:,n+1] = kx[:,n] - dt*(kx[:,n]*dudx[idys,idxs] + ky[:,n]*dvdx[idys,idxs])
            ky[:,n+1] = ky[:,n] - dt*(kx[:,n]*dudy[idys,idxs] + ky[:,n]*dvdy[idys,idxs])
            k[:,n+1] = np.sqrt(kx[:,n+1]**2+ky[:,n+1]**2)

        self.k = k
        self.kx= kx
        self.ky= ky
        self.xr= xr
        self.yr= yr
        self.theta = theta


#if __name__ == '__main__()':
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
T = 800
nt = 800
wave_period = 5
theta0 = np.pi/8
wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, theta0)
wt.set_initial_condition()
wt.solve()

fig,ax = plt.subplots(figsize=(16,6))
pc=ax.pcolormesh(wt.x,wt.y,wt.U)
for i in range(5,wt.ny-1,5):
    ax.plot(wt.xr[i,:],wt.yr[i,:],'-k')

idts = np.arange(0,wt.nt,40)
#ax.plot(wt.xr[:,idts],wt.yr[:,idts],'--k')
cb = fig.colorbar(pc)
plt.show()
