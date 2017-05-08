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
datafiles= glob.glob("../../CMSSW_8_0_4/src/Delphes/"+shift+ ".npz")#"/user_data/nelson/CMSSW_8_0_4/src/Delphes/QCDnumpy/QCD_HT500to700_1_Grid.npz")# ) ##

entries=[]
signal=[]
jet_pt=[]
jet_eta=[]
jet_phi=[]
jet_mass=[]
tau21=[]
PCeta=[]
PCphi=[]
SubEta=[]
SubPhi=[]


#print datafiles
for d in datafiles:
    data=np.load(d)
    print d
    print data['image'].shape


    pt=data['jet_pt']
    eta=data['jet_eta']
    phi=data['jet_phi']
    mass=data['jet_mass']
    sig=data['signal']
    tau=data['tau21']
    im=data['image']
    pceta=data['PCEta']
    pcphi=data['PCPhi']
    subeta=data['SubLeadingEta']
    subphi=data['SubLeadingPhi']


    
    for k in range(len(data['image'])):
    
        """
        Preforms rotation of jet using a cubic interpolation. Calculating the angle of rotation using Subleading Jets. 
        """
        print k

        if (PCEta[k] < -10) | (PCPhi[k] < -10):
            e, p = (PCEta[k], PCPhi[k])
        else:
            e, p = (SubLeadingEta, event.JetAK8_SubLeadingPhi)
            
        if e==0:continue
    
        angle = np.arctan(p / e) + 2.0 * np.arctan(1.0)
            
        if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
            angle += -4.0 * np.arctan(1.0)

        image = flip_jet(rotate_jet(im[k], -angle, dim=25),'r')
                #e_norm = np.linalg.norm(image)   
        jet_pt.append(np.float32(pt))
        jet_eta.append(np.float32(eta))
        jet_phi.append(np.float32(phi))
        jet_mass.append(np.float32(mass))
        tau21.append(np.float32(tau21))
        if 'Wprime' in files :
            signal.append(1)
        else:            
            signal.append(0)
                    
        entries.append(image)





np.savez(output+"_rotate",image=entries,signal=signal,jet_pt=jet_pt,jet_eta=jet_eta,jet_phi=jet_phi,jet_mass=jet_mass,tau21=tau21)

#uffdtype= [('image','float32'),('signal','float'),('jet_pt','float32'),('jet_eta','float32'),('jet_phi','float32'),('jet_mass','float32'),('tau21','float32')]
#f= np.array(rot,dtype=buffdtype)
#plot_mean_jet(df).savefig('test_rotate_avg.pdf')
#p.save(output+"_rotate",df)
