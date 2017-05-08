'''
maxout-train.py

author: Luke de Oliveira (lukedeo@stanford.edu)
edited: Jovan Nelson
description: script to train a maxout net

'''

import logging
import os, sys
#from keras.models import Sequential, model_from_yaml
#from keras.layers.core import *
#from keras.layers import containers
#from keras.optimizers import *
#from keras import regularizers
#from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
#from sklearn.metrics import roc_curve, auc

from viz import *

runDir= os.getcwd()

outputDir= sys.argv[1]
val= int(sys.argv[2]) #1000000
os.system('mkdir -p ' +outputDir)
#os.system('mkdir -p'+condorDir)


LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)

# X_train_image = np.load()

log('Loading data...')
# data = np.load('../../jet-simulations/trainingset.npy')
data = np.load('../../../jetimages.npy')
log('there are {} training points'.format(data.shape[0]))
log('convering to images...')
X = data['image'].reshape((data.shape[0], 25 ** 2)).astype('float32')
y = data['signal'].astype('float32')

#print len(data['image'])
log('extracting weights...')
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

signal = (signal == 1)
background = (signal == False)


# -- calculate the weights
weights = np.ones(data.shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

log('calculating weights...')
weights[signal] = get_weights(reference_distribution, pt[signal], 
    bins=np.linspace(250, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
    bins=np.linspace(250, 300, 200))
# weights[signal] = get_weights(pt[signal != 1], pt[signal], 
#   bins=np.concatenate((
#       np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
#   )

idx = range(X.shape[0])
np.random.shuffle(idx)
X = X[idx][:val]

# -- if you want to norm
# norms = np.sqrt((X ** 2).sum(-1))
# X = (X / norms[:, None])

y = y[idx][:val]#1000000
weights = weights[idx].astype('float32')[:val]


#log('Creating fold {} for feeding...'.format(foldN))#np.savez('feed_condor_fold'+str(foldN),X,y,weights,train_ix,test_ix)#foldN+=1

#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

try:
    kf=KFold(n_splits=10)
    foldN=1
    for train_ix, test_ix in kf.split(X):
        log('Creating fold {} for feeding...'.format(foldN))
        np.savez(outputDir+'/feed_condor_fold{}'.format(foldN),X=X,y=y,weights=weights,train_ix=train_ix,test_ix=test_ix,val=val,foldN=foldN)
        foldN+=1
except KeyboardInterrupt:
        log('ended early!')
