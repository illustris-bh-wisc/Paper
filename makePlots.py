from illustrisbh import readsubfHDF5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
from sklearn import mixture
import matplotlib.mlab as ml
import scipy.optimize as spopt
from matplotlib.colors import LogNorm

rc('text', usetex=True)

def make_fig():
    fig = plt.figure(figsize=(3.5,2),dpi=400)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x',labelsize=8)
    ax.tick_params(axis='y',labelsize=8)
    return fig

def save_fig(name):
    fig.savefig(name,dpi=400,bbox_inches='tight',pad_inches=0.02)

axisLabelTextSize = 9

# H_0 = 70.4 km/s/Mpc
massConversionFactor = 4.3e10 #1e10 / 0.704

# Read catalog
basedir = 'Illustris-3'
snapid = 135
catalog = readsubfHDF5.subfind_catalog(basedir,snapid)

goodMassMask = (catalog.GroupBHMass > 0) * (catalog.GroupBHMdot > 0)

# Convert to physical units (M_sun, M_sun/s)
bm_phys = massConversionFactor * catalog.GroupBHMass[goodMassMask]
bmdot_phys = 9.72e-7 * catalog.GroupBHMdot[goodMassMask]

fig = make_fig()
x = np.log10(bm_phys)
y = np.log10(bmdot_phys)
plt.scatter(x,y, marker='.', alpha=0.1)
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())
plt.ylabel(r'$\log(\dot{M}_{BH} [M_{\odot}\,s^{-1}])$',fontsize=axisLabelTextSize)
plt.xlabel(r'$\log(M_{BH} [M_{\odot}])$',fontsize=axisLabelTextSize)
save_fig('Figures/Illustris2_bhpop_full.png')

fig = make_fig()
clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(np.log10(bmdot_phys))
m1, m2 = clf.means_
w1, w2 = clf.weights_
c1, c2 = clf.covars_

histdist = plt.hist(np.log10(bmdot_phys), 100, normed=True)
plotgauss1 = lambda x: plt.plot(x,w1*ml.normpdf(x,m1,np.sqrt(c1))[0], linewidth=2)
plotgauss2 = lambda x: plt.plot(x,w2*ml.normpdf(x,m2,np.sqrt(c2))[0], linewidth=2)
plotgauss1(histdist[1])
plotgauss2(histdist[1])
plt.xlim(histdist[1].min(),histdist[1].max())
plt.ylim(histdist[0].min(),histdist[0].max())
plt.xlabel('Accretion Rate [$log(M_{\odot}\,s^{-1})$]', size=axisLabelTextSize)
plt.ylabel('fraction of sample', size=axisLabelTextSize)
save_fig('Figures/Illustris2_bhpop_mdot.png')

print "m2 = %.3f" % m2

# Make a cutoff at the mean of the second component
goodFitMassMask = np.log10(bmdot_phys) > m2

bmdot_phys = bmdot_phys[goodFitMassMask]
bm_phys = bm_phys[goodFitMassMask]
p, residual, rank, singular_values, rcond = np.polyfit( np.log10(bm_phys), np.log10(bmdot_phys), 1, full = True )
relation_params = p

print "Final sample of BHs = %d" % bm_phys.shape[0]
print "Min BH mass = %.3E" % bm_phys.min()
print "Max BH mass = %.3E" % bm_phys.max()
print "M+Mdot fit parms = " , p

fig = make_fig()
#plt.plot(np.linspace(5., 11., 100), np.polyval(p, np.linspace(5., 11., 100)), c = 'g', linestyle = '--')
H, xedges, yedges = np.histogram2d(np.log10(bm_phys), np.log10(bmdot_phys), 30)
plt.imshow(H.T, origin='lower', extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap='Oranges', norm=LogNorm(), aspect='auto', interpolation='none')
cbar = plt.colorbar(shrink=0.8, pad=0.03)
cbar.ax.tick_params(labelsize=6)
plt.ylabel(r'$log(\dot{M}_{BH} [M_{\odot}\,s^{-1}]$)', fontsize=axisLabelTextSize)
plt.xlabel(r'$log(M_{BH} [M_{\odot}])$', fontsize=axisLabelTextSize)
save_fig('Figures/Illustris2_bhpop_hist2d.png')


# Plot M_BH vs M_gas for each subhalo
groupGasMass = catalog.GroupMassType[:,0]
groupGasMass = groupGasMass[goodMassMask]
groupGasMass = massConversionFactor * groupGasMass[goodFitMassMask]

groupMass = massConversionFactor * catalog.GroupMass[goodMassMask]
groupMass = groupMass[goodFitMassMask]

fig = make_fig()
x = np.log10(groupGasMass / groupMass) + 2
y = np.log10(bmdot_phys)
z = np.log10(bm_phys)
plt.scatter(x,y, c=z, alpha=0.5, marker='.', s=3,linewidths=0, cmap='brg_r')
cbar = plt.colorbar(shrink=0.8, pad=0.03)
cbar.set_label(r'$\log(M_{BH} [M_{\odot}])$',fontsize=8)
cbar.ax.tick_params(labelsize=6)
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())
plt.xticks(np.linspace(-0.5,1.5,5),['{0:.1f}\%'.format(x) for x in 10**np.linspace(-0.5,1.5,5)])
plt.ylabel(r'$\log(\dot{M}_{BH}\,[M_{\odot}\,s^{-1}])$',fontsize=axisLabelTextSize)
plt.xlabel('$M_{gas}/M_{halo}$',fontsize=axisLabelTextSize)
save_fig('Figures/Mdot_vs_GasFrac.png')

fig = make_fig()
x = np.log10(groupGasMass)
y = np.log10(groupMass)
z = np.log10(bm_phys)
plt.scatter(x,y, c=z, alpha=0.5, marker='.',s=7,linewidths=0, cmap='Dark2')
cbar = plt.colorbar(shrink=0.8, pad=0.03)
cbar.set_label(r'$\log(M_{BH} [M_{\odot}])$',fontsize=8)
cbar.ax.tick_params(labelsize=6)
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())
plt.ylabel(r'$\log(M_{group} [M_{\odot}])$',fontsize=axisLabelTextSize)
plt.xlabel(r'$\log(M_{gas} [M_{\odot}])$',fontsize=axisLabelTextSize)
save_fig('Figures/Mgroup_vs_Mgas.png')

# finding q and K
alpha = 4.648e19
epsilon = 0.8663
eta = 0.2145

def thin_disk_approx(m_mdot, q, k):
    m = m_mdot[0]
    mdot = m_mdot[1]
    return epsilon * np.log10(mdot*alpha) + eta - np.log10(m) - q*np.log10(mdot) - k

m_mdot = [bm_phys, bmdot_phys]
bm_phys = np.log10(bm_phys)
bmdot_phys = np.log10(bmdot_phys)

fig = make_fig()
opt, cov = spopt.curve_fit(thin_disk_approx, m_mdot, np.zeros(len(bm_phys)))
x = bm_phys + opt[0] * bmdot_phys + opt[1]
y = epsilon*(bmdot_phys + np.log10(alpha)) + eta
plt.scatter(x,y, marker='.', color='b', alpha=0.2, linewidths=0, s=7)
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())

# "interpolate" m and mdot onto a finer line
_m = np.linspace(np.min(bm_phys),np.max(bm_phys),200)
_mdot = np.linspace(np.min(bmdot_phys),np.max(bmdot_phys),200)

# Use that finer line to draw the fit
x = _m + opt[0]*_mdot + opt[1]
y = epsilon*(_mdot + np.log10(alpha)) + eta
plt.plot(x,y, linestyle='--', c='r', linewidth=2)
plt.ylabel(r'$\log\,L_{x}$',fontsize=axisLabelTextSize)
plt.xlabel(r"$\log M+{0:.3}\log\dot{{M}}+{1:.3}$".format(opt[0],opt[1]),fontsize=axisLabelTextSize)
save_fig('Figures/fp_fit.png')


# Plot the elvis data
data = np.loadtxt('elvis_templates.csv',delimiter=',')

L_bol = data[:,0]-np.log10(3.846e33)
L_xray = data[:,1]-np.log10(3.846e33)
p = np.poly1d(np.polyfit(L_bol,L_xray,1))
fig = make_fig()
plt.scatter(L_bol,L_xray, marker='+', c='blue')
plt.plot(L_bol,p(L_bol),'r-')
plt.xlim(L_bol.min(),L_bol.max()*1.002)
plt.ylim(L_xray.min(),L_xray.max()*1.002)
plt.ylabel(r'$\log(L_{x} [L_{\odot}])$', size=axisLabelTextSize)
plt.xlabel(r'$\log(L_{bol} [L_\odot])$', size=axisLabelTextSize)
save_fig('Figures/elvis_template.png')
