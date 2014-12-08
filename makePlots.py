from illustrisbh import readsubfHDF5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys
import emcee
from matplotlib.colors import LogNorm
from sklearn import mixture
import matplotlib.mlab
import scipy.optimize as spopt

rc('text', usetex=True)

def make_fig():
	fig = plt.figure(figsize=(3.5,2),dpi=400)
	ax = fig.add_subplot(111)
	ax.tick_params(axis='x',labelsize=8)
	ax.tick_params(axis='y',labelsize=8)
	return fig

def save_fig(name):
	fig.savefig(name,dpi=400,bbox_inches='tight',pad_inches=0.0)


axisLabelTextSize = 9

#data = np.loadtxt('elvis_templates.csv',delimiter=',')

#L_bol = data[:,0]*1e14
#L_xray = data[:,1]*1e14
#p = np.poly1d(np.polyfit(L_bol,L_xray,1))
#fig = make_fig()
#plt.scatter(L_bol,L_xray, marker='+', c='blue')
#plt.plot(L_bol,p(L_bol),'r-')
#plt.xlim(L_bol.min(),L_bol.max())
#plt.ylim(L_xray.min(),L_xray.max())
#plt.ylabel(r'X-ray Luminosity [$10^{-14}\,L_\odot$]', size=axisLabelTextSize)
#plt.xlabel(r'Bolometric Luminosity [$10^{-14}\,L_\odot$]', size=axisLabelTextSize)
#save_fig('Figures/elvis_template.png')

# Read catalog
basedir = 'Illustris-3'
snapid = 135
catalog = readsubfHDF5.subfind_catalog(basedir,snapid)

cond = (catalog.GroupBHMass != 0) * (catalog.GroupBHMdot > 0)

# Convert to physical units (M_sun, M_sun/yr)
bm_phys = 4.3e10 * catalog.GroupBHMass[cond]
bmdot_phys = 9.72e-7 * 3.14e7 * catalog.GroupBHMdot[cond]

#fig = make_fig()
#x = np.log10(bm_phys)
#y = np.log10(bmdot_phys)
#plt.scatter(x,y, marker='.', alpha=0.1)
#plt.xlim(x.min(),x.max())
#plt.ylim(y.min(),y.max())
#plt.ylabel(r'$\log(\dot{M}_{BH} [M_{\odot}\,yr^{-1}])$',fontsize=axisLabelTextSize)
#plt.xlabel(r'$\log(M_{BH} [M_{\odot}])$',fontsize=axisLabelTextSize)
#save_fig('Figures/Illustris2_bhpop_full.png')

fig = make_fig()
clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(np.log10(bmdot_phys))
m1, m2 = clf.means_
w1, w2 = clf.weights_
c1, c2 = clf.covars_

#histdist = matplotlib.pyplot.hist(np.log10(bmdot_phys), 100, normed=True)
#plotgauss1 = lambda x: plt.plot(x,w1*matplotlib.mlab.normpdf(x,m1,np.sqrt(c1))[0], linewidth=2)
#plotgauss2 = lambda x: plt.plot(x,w2*matplotlib.mlab.normpdf(x,m2,np.sqrt(c2))[0], linewidth=2)
#plotgauss1(histdist[1])
#plotgauss2(histdist[1])
#plt.xlim(-22,1)
#plt.ylim(0,0.55)
#plt.xlabel('Accretion Rate [$log(M_{\odot} yr^{-1})$]', size=axisLabelTextSize)
#plt.ylabel('fraction of sample', size=axisLabelTextSize)
#save_fig('Figures/Illustris2_bhpop_mdot.png')

print "m2 = %.3f" % m2

bmdot_phys = bmdot_phys[ np.log10(bmdot_phys) > m2]
bm_phys = bm_phys[ np.log10(bmdot_phys) > m2 ]
p, residual, rank, singular_values, rcond = np.polyfit( np.log10(bm_phys), np.log10(bmdot_phys), 1, full = True )
relation_params = p

print "Final sample of BHs = %d" % bm_phys.shape[0]
print "Min BH mass = %.3E" % bm_phys.min()
print "Max BH mass = %.3E" % bm_phys.max()
print "M+Mdot fit parms = " , p

#fig = make_fig()
#plt.plot(np.linspace(5., 11., 100), np.polyval(p, np.linspace(5., 11., 100)), c = 'g', linestyle = '--')
#H, xedges, yedges = np.histogram2d(np.log10(bm_phys), np.log10(bmdot_phys), 30)
#plt.imshow(H.T, origin='lower', extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap='Oranges', norm=LogNorm(), aspect='auto', interpolation='none')
#cbar = plt.colorbar(shrink=0.8, pad=0.03)
#cbar.ax.tick_params(labelsize=6)
#plt.ylabel(r'$log(\dot{M}_{BH} [M_{\odot}\,yr^{-1}]$)', fontsize=axisLabelTextSize)
#plt.xlabel(r'$log(M_{BH} [M_{\odot}])$', fontsize=axisLabelTextSize)
#save_fig('Figures/Illustris2_bhpop_hist2d.png')

## finding q and K using convergence
# General idea:
# 
# For Lx ~ M
# Define function f(M,q,k) = log10(a + b/M) - 0.694 q log10(M) + 16.3769 - k = 0
# a = 623.04
# b = 1.656e-15
# find zeros
# 
# For Lx ~ \.{M}
# Define function f(\.{M},q,k) = log10(a + b/\.{M}) - 0.694 q log10(\.{M}(M)} + 16.3769 - k = 0
# a = 9.03e18
# b = 1.656e-15
# \.{M}(M) is given by the orange plot
# 
# solve system of equations:
# 
# log10(a1 + b/M) - d*q*log10(M) + e*q - log10(a2*Mdot + b) + 1/d * log10(Mdot) - e/d * q * log10(Mdot) = 0
def f(q, m, mdot):
    a1 = 623.04
    b = 1.656e-15
    d = relation_params[0]
    a2 = 9.03e18
    e = -relation_params[1]
    return np.log10(a1 + b/m) - d*q*np.log(m) + e*q - np.log10(a2*mdot + b) + (1. - e*q)/d * np.log10(mdot)

sols = []
for i in range(len(bmdot_phys)):
    sol = spopt.root(f, [1.], args = (bm_phys[i], (1/3.14e7)*bmdot_phys[i]))
    sols.append(sol.x[0])

fig = make_fig()
(n,bins,_) = plt.hist(sols, bins=50, log=True)
plt.axvline(np.mean(sols), c = 'r', linewidth = 3, linestyle = '--')
plt.xlim(bins.min(),bins.max())
plt.ylim(n.min(),n.max())
plt.xlabel(r'best-fit $q$ value', size=axisLabelTextSize)
plt.ylabel('$N_{BH}$',size=axisLabelTextSize)
save_fig('Figures/q_nr_hist.png')
