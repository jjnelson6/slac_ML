###The Purpose of the file is to combine the network outputs from K-folds. Use this when training and testing a small set of data#### 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#ljmetDir = '/mnt/hadoop/users/nelson/TreeGrids/'
yhfiles = '../../../training_output/60k*-200PU*.npy'#'/mnt/hadoop/users/nelson/GridTree/'+shift+'*'+t
jetinfofiles='../../../training_output/jet_info_200PU*.npz'

outputnpy='../../../training_output/train200PU_test200PU'
outputnpz='../../../Jetimages_200PU'

#intialze storing lists
yhat=[]
jet_pt=[]
jet_eta=[]
jet_phi=[]
jet_mass=[]
signal=[]
tau21=[]
image=[]
yhdata=glob.glob(yhfiles)
jetinfo=glob.glob(jetinfofiles)

count=0
#print (yhdata)
#print (jetinfo)

for yfiles in yhdata:
    
    if len(jetinfo)!=len(yhdata):
        print( " The number of Jet Info files do not match the number of yh files!!")
        break
    
    print ('DNN output File:',yfiles)
    print ('Jet info File:',jetinfo[count])
    
    y=np.load(yfiles)
    jets=np.load(jetinfo[count])
    
    pt=jets['jet_pt']
    eta=jets['jet_eta']
    phi=jets['jet_phi']
    mass=jets['jet_mass']
    sig=jets['signal']
    tau=jets['tau21']
    im=jets['image']
   
    for lon in range(len(y)):
        if len(y)!=len(pt):
            print ("Length of DNN output does not match with the length of Jetinfo!!")
            break
        
        yhat.append(y[lon])
        jet_pt.append(pt[lon])
        jet_eta.append(eta[lon])
        jet_phi.append(phi[lon])
        jet_mass.append(mass[lon])
        signal.append(sig[lon])
        tau21.append(tau[lon])
        image.append(im[lon])

    count+=1



np.save(outputnpy,yhat)#saving combined DNN output
np.savez(outputnpz,image=image,signal=signal,jet_pt=jet_pt, jet_eta=jet_eta, jet_phi=jet_phi, jet_mass=jet_mass,tau21=tau21) #saving jet information

                
                       

