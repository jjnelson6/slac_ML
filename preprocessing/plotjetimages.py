import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math as m
shift=sys.argv[1]
output= '../../Jet_images/'+shift

def plot_mean_jet(rec, field = 'image', title = 'Average Jet Image',signal=1):
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec[field][rec['signal']==signal], axis = 0),aspect='auto' ,norm=LogNorm(vmin=.001, vmax=10), extent=(min(rec['jet_eta']),max(rec['jet_eta']),min(rec['jet_phi']),max(rec['jet_phi'])),interpolation='nearest',cmap='jet')
    plt.colorbar(im)
    plt.xlabel(r'[Translated] Psuedorapidity ($\eta$)')
    plt.ylabel(r'[Translated] Azimuthal Angle $\phi$')
    #plt.rc('font',size=37)
    #plt.rc('figure', titlesize=35)
    #plt.rc('legend',fontsize=38)
    plt.title(r''+title)
    return fig


def plot_jet(rec, title = 'Jet Image', log=True):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    if log:
        ax.imshow(rec[0],  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
        ax.imshow(rec[1],  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
        ax.imshow(rec[2],  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    else:
        im = ax.imshow(rec, interpolation='nearest')
    plt.title(title)
    return fig



data=np.load(shift+".npz")
plot_mean_jet(data,title=r"$p_T \in [200, 300]$ GeV, $m_{\mathsf{jet}}\in [65, 95]$ GeV"+ '\n'+ r"Delphes  0PU:  $W' \rightarrow WZ$").savefig(output+'_Wprime.jpg')
plot_mean_jet(data,signal=0,title=r"$p_T \in [200, 300]$ GeV, $m_{\mathsf{jet}}\in [65, 95]$ GeV"+'\n'+ "Delphes 0PU: QCD").savefig(output+'_QCD.jpg')


