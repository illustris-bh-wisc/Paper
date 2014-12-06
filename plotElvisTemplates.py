import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

data = np.loadtxt('elvis_templates.csv',delimiter=',')

L_bol = data[:,0]*1e14
L_xray = data[:,1]*1e14

p = np.poly1d(np.polyfit(L_bol,L_xray,1))

fig = plt.figure(figsize=(6,4), dpi=400)
plt.scatter(L_bol,L_xray, marker = '.', c = 'grey')
plt.plot(L_bol,p(L_bol),'r-')
plt.xlim(L_bol.min(),L_bol.max())
plt.ylim(L_xray.min(),L_xray.max())
plt.ylabel(r'X-ray Luminosity [$10^{-14}\,L_\odot$]', size=12)
plt.xlabel(r'Bolometric Luminosity [$10^{-14}\,L_\odot$]', size=12)
fig.savefig('Figures/elvis_template.png')
