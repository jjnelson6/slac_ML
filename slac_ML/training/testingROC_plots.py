#from keras.models import Sequential, model_from_yaml
#from keras.layers.core import Dense, Dropout, MaxoutDense, Activation, Merge # AutoEncoder
#from keras.layers.advanced_activations import PReLU
#from keras.layers.embeddings import Embedding
#from keras.layers.noise import GaussianNoise
#from keras.optimizers import SGD, RMSprop, Adagrad, Adam
#from keras import regularizers
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
#import matplotlib.pyplot as plt 
#from matplotlib.colors import LogNorm
from viz import *
from likelihood import *

PLOT_DIR = '/user_data/nelson/training_output/'
network_eval= 'train0PU_test200PU.npy'
data = np.load('../../../Jetimages_200PU.npz')

signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau21']

signal_pct = data['signal'].mean()
#print '{}% signal'.format(signal_pct)

signal = (signal == 1)
background = (signal == False)

# -- calculate the weights
weights = np.ones(data['image'].shape[0])

reference_distribution = pt[background]

#re-weights a actual distribution to a target.  
weights[signal] = get_weights(reference_distribution, pt[signal], 
	bins=np.linspace(200, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
	bins=np.linspace(200, 300, 200))

y_dl= np.load(network_eval)

#Plots Curves
discs = {}
add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal, 2-tau_21, weights=weights), discs)
add_curve(r'Deep Network, test-trained on $p_T \in [250, 300]$ GeV', 'red', calculate_roc(signal, y_dl, weights=weights, bins=1000000), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD ($p_T \in [250, 300]$ GeV, matched to QCD)" + '\n' + 
	r'$m_{\mathsf{jet}}\in [65, 95]$ GeV', min_eff = 0.2, max_eff=0.8, logscale=False)
fg.savefig(PLOT_DIR+'0PUtrain_200PUtest.png')
plt.show()
