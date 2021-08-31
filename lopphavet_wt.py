import numpy as np
import matplotlib.pyplot as plt
from wave_tracing_FE import Wave_tracing_FE
import xarray as xa
import cmocean

idt=0 #22
idx0 = 0
u_eastwards = xa.open_dataset('current_forcing/u_eastward_lopphavet.nc')
v_northwards = xa.open_dataset('current_forcing/v_northward_lopphavet.nc')

data_time = str(u_eastwards.isel(time=idt).time.data).split(':')[0]

X = u_eastwards.X#[idx0:]
Y = u_eastwards.Y
nx = len(X)#U.shape[1]
ny = len(Y)#U.shape[0]
dx=dy=800
nb_wave_rays = 150#330 #350

T = 22000
print("T={}h".format(T/3600))
nt = 12000#15000
wave_period = 10
central_angle = 153#148#149
theta0 = [
          -(central_angle*np.pi)/180]#,-(55*np.pi)/180] #np.pi/10
X0, XN = X[0].data, X[-1].data
Y0, YN = Y[0].data, Y[-1].data

HM = []
for th0 in theta0:
    wt = Wave_tracing_FE(u_eastwards.u_eastward,v_northwards.v_northward,
                        nx, ny, nt,T,dx,dy, wave_period, th0,nb_wave_rays=nb_wave_rays,
                        domain_X0=X0, domain_XN=XN,
                        domain_Y0=Y0, domain_YN=YN, temporal_evolution=True, incoming_wave_side='top')
    wt.set_initial_condition()
    wt.solve()

### PLOTTING
# Georeference
proj4='+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70' #NK800
lons,lats=wt.to_latlon(proj4)

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# read wam data:
model_period = '20190201'
wam_diff = xa.open_dataset('/lustre/storeB/users/trygveh/data/moskenes/wam/wam_diff_{}00.nc'.format(model_period), decode_cf=True)
wam_coord = xa.open_dataset('/lustre/storeB/users/trygveh/data/moskenes/wam/TRUEcoordDepthc1exte.nc')

idt = 405

wam_lat = wam_coord.latitude
wam_lon = wam_coord.longitude

hs_diff_max = .75#np.abs(wam_diff.hs[idt,:,:]).max()


fig2, ax2 = plt.subplots(frameon=False,figsize=(8,10),subplot_kw={'projection': ccrs.Mercator(), 'facecolor':"gray"})

cf2=ax2.pcolormesh(wam_lon, wam_lat, wam_diff.hs[idt,:,:], cmap=cmocean.cm.balance,
                       transform=ccrs.PlateCarree(),vmin=-hs_diff_max,vmax=hs_diff_max)

for i in range(0,wt.nb_wave_rays,1):
    ax2.plot(lons[i,:],lats[i,:],'-',c='black',transform=ccrs.PlateCarree(),alpha=0.3)

ax2.coastlines()
ax2.set_extent([19.7, 22.1, 69.9, 70.8], crs=ccrs.PlateCarree())
#fig2.savefig('figures/ray_tracing_lopphavet.png',dpi=180, transparent=False)
plt.show()

"""
fs = 20
land_mask = np.ma.ones(UU[0].shape)
land_mask.mask=False
land_mask.mask[~np.isnan(UU[0])]=True # is not nan
#speed = np.sqrt(U**2 + V**2)
#vorticity = wt.dvdx-wt.dudy

# coriolis
latitude_coriolis =  67.8
Omega = 7.2921*1e-5 # rad/s
latitude_coriolis_rad = np.deg2rad(latitude_coriolis)
inertial = 2*Omega*np.sin(latitude_coriolis_rad)

#plot_X0, plot_XN = X0, 68.2*1e4
#plot_Y0, plot_YN = 105*1e4, 130*1e4
plot_X0, plot_XN = X0, 68.2*1e4
plot_Y0, plot_YN = 159*1e4, 130*1e4

fig,ax = plt.subplots(figsize=(6,8))
for i in range(0,wt.nb_wave_rays):
    ax.plot((wt.xr[i,:]-plot_X0)/1e3,(wt.yr[i,:]-plot_Y0)/1e3,'-k')
ax.pcolormesh((X-plot_X0)/1e3,(Y-plot_Y0)/1e3,land_mask,shading='auto')

ax.set_xlim([0,80])
ax.set_ylim([0,100])

fig.tight_layout()
ax.tick_params(axis='both',labelsize=fs-4)
#fig.savefig('figures/ray_tracing_lopphavet.png',dpi=180, transparent=True)

plt.show()


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
