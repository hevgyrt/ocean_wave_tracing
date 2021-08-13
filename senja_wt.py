import numpy as np
import matplotlib.pyplot as plt
from wave_tracing_FE import Wave_tracing_FE
import xarray as xa
import cmocean

idt=5 #22
idx0 = 100
u_eastwards = xa.open_dataset('current_forcing/u_eastward_senja.nc')
v_northwards = xa.open_dataset('current_forcing/v_northward_senja.nc')
data_time = str(u_eastwards.isel(time=idt).time.data).split(':')[0]
U = np.fliplr(u_eastwards.isel(time=idt).to_array()[0,:-idx0,:].data.T)
V = np.fliplr(v_northwards.isel(time=idt).to_array()[0,:-idx0,:].data.T)

X = u_eastwards.Y[idx0:]
Y = u_eastwards.X
nx = len(X)#U.shape[1]
ny = len(Y)#U.shape[0]
#nx = 100
#ny=40
dx=dy=800
nb_wave_rays = 150#330 #350

T = 20000
print("T={}h".format(T/3600))
nt = 3000
wave_period = 8
#theta0 = [(2.5*np.pi)/180, (-2.5*np.pi)/180, (5*np.pi)/180, (-5*np.pi)/180,0] #np.pi/10
central_angle = 53#53
theta0 = [#-((central_angle+2.5)*np.pi)/180,-((central_angle+1.25)*np.pi)/180,
          #-((central_angle-2.5)*np.pi)/180,-((central_angle-1.25)*np.pi)/180,
          -(central_angle*np.pi)/180]#,-(55*np.pi)/180] #np.pi/10
X0, XN = X[0].data, X[-1].data
Y0, YN = Y[0].data, Y[-1].data

HM = []
for th0 in theta0:
    wt = Wave_tracing_FE(U, V, nx, ny, nt,T,dx,dy, wave_period, th0,nb_wave_rays=nb_wave_rays,
                        domain_X0=X0, domain_XN=XN,
                        domain_Y0=Y0, domain_YN=YN, temporal_evolution=False)
    wt.set_initial_condition()
    wt.solve()
    xx,yy,hm = wt.ray_density(3,5)

### PLOTTING
import matplotlib.colors as colors

fs = 20
land_mask = np.ma.ones(U.shape)
land_mask.mask=False
land_mask.mask[~np.isnan(U)]=True # is not nan
speed = np.sqrt(U**2 + V**2)
vorticity = wt.dvdx-wt.dudy

# coriolis
latitude_coriolis =  67.8
Omega = 7.2921*1e-5 # rad/s
latitude_coriolis_rad = np.deg2rad(latitude_coriolis)
inertial = 2*Omega*np.sin(latitude_coriolis_rad)

#plot_X0, plot_XN = X0, 68.2*1e4
#plot_Y0, plot_YN = 105*1e4, 130*1e4
plot_X0, plot_XN = X0, 68.2*1e4
plot_Y0, plot_YN = 144*1e4, 130*1e4

fig,ax = plt.subplots(figsize=(6,8))
for i in range(0,wt.nb_wave_rays):
    ax.plot((wt.xr[i,:]-plot_X0)/1e3,(wt.yr[i,:]-plot_Y0)/1e3,'-k')
ax.pcolormesh((X-plot_X0)/1e3,(Y-plot_Y0)/1e3,land_mask,shading='auto')

ax.set_xlim([0,90])
ax.set_ylim([0,130])

fig.tight_layout()
ax.tick_params(axis='both',labelsize=fs-4)
#fig.savefig('figures/ray_tracing_senja_{}.png'.format(data_time),dpi=180, transparent=True)

plt.show()


proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
lons,lats=wt.to_latlon(proj4, flip_xy=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
fig, ax = plt.subplots(frameon=False,figsize=(7,7),subplot_kw={'projection': ccrs.Mercator()})

for i in range(wt.nb_wave_rays):
    ax.plot(lons[i,:],lats[i,:],'-k',transform=ccrs.PlateCarree())

# Deciding extent
lonmin, lonmax, latmin, latmax = 7,15.0,65,72
ax.set_extent([lonmin, lonmax, latmin, latmax]) #x0, x1, y0, y1) of the map in the given coordinate system.

# adding coastline

lscale = 'auto' # Scale for coasline
f = cfeature.GSHHSFeature(scale=lscale, levels=[1],
        facecolor=cfeature.COLORS['land'])
ax.add_geometries(
        f.intersecting_geometries([lonmin, lonmax, latmin, latmax]),
        ccrs.PlateCarree(),
        facecolor=cfeature.COLORS['land_alt1'],
        edgecolor='black')

plt.show()






"""
fig,ax = plt.subplots(figsize=(8,10))

plot_X0, plot_XN = X0, 68.2*1e4
plot_Y0, plot_YN = 105*1e4, 130*1e4
pc=ax.pcolormesh((X-plot_X0)/1e3,(Y-plot_Y0)/1e3,vorticity/inertial, cmap=cmocean.cm.balance,vmin=-2.9,vmax=2.9)
for i in range(0,wt.nb_wave_rays):
    ax.plot((wt.xr[i,:]-plot_X0)/1e3,(wt.yr[i,:]-plot_Y0)/1e3,'-k')

ax.set_xlim([0,145])
ax.set_ylim([0,270])
cb = fig.colorbar(pc,extend='both',shrink=0.95, pad=0.02)
cb.ax.set_title(r'$\zeta / f$',fontsize=fs)
fig.tight_layout()
ax.tick_params(axis='both',labelsize=fs-4)
cb.ax.tick_params(labelsize=fs)
#fig.savefig('ray_tracing_{}.png'.format(data_time),dpi=160, transparent=True)
plt.show()


xx,yy, hm = wt.ray_density(10,10)
plt.pcolormesh(xx,yy,hm)
plt.colorbar()
for i in range(0,nb_wave_rays):
    plt.plot(wt.xr[i,:],wt.yr[i,:],'-r',alpha=0.3)
plt.scatter(xx,yy);plt.show()
"""
