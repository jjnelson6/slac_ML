import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import math as m
from jettools import plot_mean_jet,flip_jet,rotate_jet
from processing import buffer_to_jet



def plot_mean_jet(rec, field = 'images', title = 'Average Jet Image'):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec, axis = 0),  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    plt.title(r''+title)
    return fig


data=np.load("../../CMSSW_8_0_4/src/Delphes/Wprime_all.npz")

temp= {}
temp['images'] = data['image']
temp['jet_eta']=data['jet_eta']
temp['jet_phi']= data['jet_phi']


rot=[]
for k in range(len(temp['images'])):
    e, p = (temp['jet_eta'][k], temp['jet_phi'][k])
    
    angle = np.arctan(p / e) + 2.0 * np.arctan(1.0)

    if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
        angle += -4.0 * np.arctan(1.0)

    image = flip_jet(rotate_jet(np.array(temp['images'][k]), -angle, normalizer=4000.0, dim=25),'r')
    e_norm = np.linalg.norm(image)
   
    if e_norm>0:
        rot.append((image).astype('float32'))

#print rot.shape



#buff=buffer_to_jet(,1,max_entry=100000, pix=25)
    #rot.append(buff)



buffdtype = [('images', 'float32')]



df= np.array(rot,dtype=buffdtype)
print df.shape
#plot_mean_jet(df).savefig('test_rotate_avg.pdf')


np.save('rotate_Wprime',df)
