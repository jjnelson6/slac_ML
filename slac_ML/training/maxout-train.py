'''
maxout-train.py

author: Luke de Oliveira (lukedeo@stanford.edu)
edited: Jovan Nelson
description: script to train a maxout net

'''
import logging
from keras.models import Sequential, model_from_yaml
from keras.layers.core import *
from MaxoutDense import *
from keras.optimizers import *
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
from sklearn.metrics import roc_curve, auc

from viz import *

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)

class ROCModelCheckpoint(Callback):
    '''
    Callback to optimize AUC
    '''
    def __init__(self, filepath, X, y, weights, verbose=True):
        super(Callback, self).__init__()

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.filepath = filepath
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        yh = self.model.predict(self.X, verbose=True).ravel()
        fpr, tpr, _ = roc_curve(self.y, yh, sample_weight=self.weights)
        select = (tpr > 0.1) & (tpr < 0.9)
        current = auc(tpr[select], 1 / fpr[select])
        
        if current > self.best:
            if self.verbose > 0:
                print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                      % (epoch, 'AUC', self.best, current, self.filepath))
                self.best = current
            self.model.save_weights(self.filepath, overwrite=True)
            np.save(self.filepath.replace('.h5',''), yh.astype('float32'))
        else:
            if self.verbose > 0:
                print("Epoch %05d: %s did not improve" % (epoch, 'AUC'))


# X_train_image = np.load()

log('Loading data...')
# data = np.load('../../jet-simulations/trainingset.npy')
data = np.load('../../../Jetimages_140PU.npz')
log('there are {} training points'.format(data['image'].shape[0]))
log('convering to images...')
X = data['image'].reshape((data['image'].shape[0], 25 ** 2)).astype('float32')
y = data['signal'].astype('float32')

num_inputs= 210000 ###1000000

log('extracting weights...')
signal, pt, mass, tau_21  = data['signal'], data['jet_pt'], data['jet_mass'], data['tau21']

signal = (signal == 1)
background = (signal == False)

# -- calculate the weights
weights = np.ones(data['image'].shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

log('calculating weights...')
weights[signal] = get_weights(reference_distribution, pt[signal], 
    bins=np.linspace(200, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
    bins=np.linspace(200, 300, 200))

idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx][:num_inputs]

# -- if you want to norm
# norms = np.sqrt((X ** 2).sum(-1))
# X = (X / norms[:, None])

y = y[idx][:num_inputs]#1000000
weights = weights[idx].astype('float32')[:num_inputs]

sig=data['signal'][idx][:num_inputs]
jet_pt=pt[idx][:num_inputs]
jet_mass=mass[idx][:num_inputs]
jet_eta=data['jet_eta'][idx][:num_inputs]
jet_phi=data['jet_phi'][idx][:num_inputs]
tau_21=tau_21[idx][:num_inputs]
image= data['image'][idx][:num_inputs]

from sklearn.model_selection import KFold
try:
    kf = KFold(n_splits=3)#10
    foldN = 1
    for train_ix, test_ix in kf.split(X):
        log('Working on fold: {}'.format(foldN))

        log('Building new submodel...')
       # -- build the model
        dl = Sequential()
        dl.add(MaxoutDense(256, 5, input_shape=(625, ), init='he_uniform'))
        dl.add(Dropout(0.3))

        dl.add(MaxoutDense(128, 5, init='he_uniform'))
        dl.add(Dropout(0.2))

        dl.add(Dense(64))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.2))

        dl.add(Dense(25))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.3))

        dl.add(Dense(1))
        dl.add(Activation('sigmoid'))

        log('compiling...')

        dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        log('training!')
        
        h = dl.fit(X[train_ix], y[train_ix], batch_size=32, nb_epoch=50,validation_data=(X[test_ix], y[test_ix]),callbacks = 
    	               [
                EarlyStopping(verbose=False, patience=10, monitor='val_loss'),
                ModelCheckpoint('../../../training_output/70k-140PU-SLACNormalized-final-logloss-cvFold{}.h5'.format(foldN), monitor='val_loss', verbose=False, save_best_only=True),
                ROCModelCheckpoint('../../../training_output/70k-140PU-SLACNormalized-final-roc-cvFold{}.h5'.format(foldN), X[test_ix], y[test_ix], weights[test_ix], verbose=True)
    	               ],
                       sample_weight=weights[train_ix]
                )
        
        #np.savez('../../../training_output/jet_info_140PUkfold{}'.format(foldN),image=image[test_ix],signal=sig[test_ix],jet_pt=jet_pt[test_ix],jet_mass=jet_mass[test_ix],tau21=tau_21[test_ix],jet_eta=jet_eta[test_ix], jet_phi=jet_phi[test_ix]) 
        foldN += 1
                      # sample_weight=np.power(weights, 0.7))
except KeyboardInterrupt:
	log('ended early!')


#with open('../../../training_output/70k-35PU-NewSLAC-final.yaml', 'wb') as f:
#	f.write(dl.to_yaml())



