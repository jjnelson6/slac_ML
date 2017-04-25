import sys,glob
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import math as m
from jettools import plot_mean_jet,flip_jet,rotate_jet
#from processing import buffer_to_jet

def plot_mean_jet(rec, field = 'image', title = 'Average Jet Image'):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec, axis = 0),  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    plt.title(r''+title)
    return fig

shift=sys.argv[1]
output=shift
#datafiles= glob.glob("../../CMSSW_8_0_4/src/Delphes/"+shift+ ".npz")#"/user_data/nelson/CMSSW_8_0_4/src/Delphes/QCDnumpy/QCD_HT500to700_1_Grid.npz")# ) ##
im=[]
signal=[]
jet_pt=[]
jet_eta=[]
jet_phi=[]
jet_mass=[]
tau21=[]
#print datafiles
#for d in datafiles:


data=np.load(shift+".npz")
print data['image'].shape

#for k in range(len(data['image'])):
    
#    print k
#    im.append(data['image'][k])
signal.append(data['signal'][0])
print signal
jet_pt.append(data['jet_pt'][0])
jet_eta.append(data['jet_eta'][0])
print jet_eta
jet_phi.append(data['jet_phi'][0])
jet_mass.append(data['jet_mass'][0])
print jet_mass
tau21.append(data['tau21'][0])
print 'got here'



np.savez(output+"_combined",image=data['image'],signal=signal,jet_pt=jet_pt,jet_eta=jet_eta,jet_phi=jet_phi,jet_mass=jet_mass,tau21=tau21)

#uffdtype= [('image','float32'),('signal','float'),('jet_pt','float32'),('jet_eta','float32'),('jet_phi','float32'),('jet_mass','float32'),('tau21','float32')]
#f= np.array(rot,dtype=buffdtype)
#plot_mean_jet(df).savefig('test_rotate_avg.pdf')
#p.save(output+"_rotate",df)
