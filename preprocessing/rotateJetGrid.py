import sys,glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math as m
from jettools import plot_mean_jet,flip_jet,rotate_jet

shift=sys.argv[1]
output=shift
datafiles= glob.glob("../../"+shift+ ".npz")

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
    print (d)
#   print (data['image'].shape)
    
    pt=data['jet_pt'][data['jet_eta']!=-99]
    eta=data['jet_eta'][data['jet_eta']!=-99]
    phi=data['jet_phi'][data['jet_eta']!=-99]
    mass=data['jet_mass'][data['jet_eta']!=-99]
    sig=data['signal'][data['jet_eta']!=-99]
    tau=data['tau21'][data['jet_eta']!=-99]
    im=data['image'][data['jet_eta']!=-99]
    pceta=data['PCEta'][data['jet_eta']!=-99]
    pcphi=data['PCPhi'][data['jet_eta']!=-99]
    subeta=data['SubLeadingEta'][data['jet_eta']!=-99]
    subphi=data['SubLeadingPhi'][data['jet_eta']!=-99]

    print ('Done Uploading Jet Info!')
    
    for k in range(len(pt)):
    
        """
        Preforms rotation of jet using a cubic interpolation. Calculating the angle of rotation using Subleading Jets. 
        """
        if k%10000==0:
            print ('Events Processed:',k)
            

        if (subeta[k] ==-99) | (subphi[k] ==-99):
            e, p = (pceta[k], pcphi[k])
        else:
            e, p = (subeta[k],subphi[k])

            
        if e==0:continue
    
        angle = np.arctan(p / e) + 2.0 * np.arctan(1.0)
            
        if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
            angle += -4.0 * np.arctan(1.0)

        image = flip_jet(rotate_jet(np.array(im[k]), -angle, dim=25),'r')
        #e_norm = np.linalg.norm(image)   
        jet_pt.append(np.float32(pt[k]))
        jet_eta.append(np.float32(eta[k]))
        jet_phi.append(np.float32(phi[k]))
        jet_mass.append(np.float32(mass[k]))
        tau21.append(np.float32(tau[k]))
        signal.append(np.float32(sig[k]))
        entries.append(image)

print('Saving Rotated Jets')
np.savez(output.replace('unrotated','rotated'),image=entries,signal=signal,jet_pt=jet_pt,jet_eta=jet_eta,jet_phi=jet_phi,jet_mass=jet_mass,tau21=tau21)
