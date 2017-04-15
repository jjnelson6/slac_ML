import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import math as m
from jettools import plot_mean_jet,flip_jet,rotate_jet
#from processing import buffer_to_jet



def plot_mean_jet(rec, field = 'images', title = 'Average Jet Image'):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec, axis = 0),  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    plt.title(r''+title)
    return fig


data=np.load("../../CMSSW_8_0_4/src/Delphes/QCD_all.npz")
print data['image'].shape

rot=[]
for k in range(len(data['image'])):
    
    """
    Preforms rotation of jet using a cubic interpolation. Calculating the angle of rotation using Subleading Jets. 
    """

    if (data['SubLeadingEta'][k] < -10) | (data['SubLeadingPhi'][k] < -10):
        e, p = (data['PCEta'][k], data['PCPhi'][k])
    else:
        e, p = (data['SubLeadingEta'][k], data['SubLeadingPhi'][k])
    
    angle = np.arctan(p / e) + 2.0 * np.arctan(1.0)

    if (-np.sin(angle) * e + np.cos(angle) * p) > 0:
        angle += -4.0 * np.arctan(1.0)

    image = flip_jet(rotate_jet(np.array(data['image'][k]), -angle, normalizer=4000.0, dim=25),'r')
    e_norm = np.linalg.norm(image)   
    
    if e_norm>0:
        rot.append((image).astype('float32'))


buffdtype = [('images', 'float32')]
df= np.array(rot,dtype=buffdtype)

#plot_mean_jet(df).savefig('test_rotate_avg.pdf')
np.save('rotate_QCD',df)
