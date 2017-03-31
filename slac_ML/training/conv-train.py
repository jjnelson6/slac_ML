'''
conv-train.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to train a conv net

'''
from keras.models import Sequential #model_from_yaml
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation, Flatten #Merge 
from N_with_LRN2D import LRN2D
#from keras.layers import containers
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
from sklearn.metrics import roc_curve, auc
from viz import *
import theano

class ROCModelCheckpoint(Callback):
    def __init__(self, filepath, X, y, weights, verbose=True):
        super(Callback, self).__init__()

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.filepath = filepath
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        fpr, tpr, _ = roc_curve(self.y, self.model.predict(self.X, verbose=True).ravel(), sample_weight=self.weights)
        select = (tpr > 0.1) & (tpr < 0.9)
        current = auc(tpr[select], 1 / fpr[select])

        if current > self.best:
            if self.verbose > 0:
                print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                      % (epoch, 'AUC', self.best, current, self.filepath))
            self.best = current
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.verbose > 0:
                print("Epoch %05d: %s did not improve" % (epoch, 'AUC'))


print 'Constructing net'
#FILTER_SIZES = [(11, 11), (3, 3), (3, 3), (3, 3)]

dl = Sequential()
dl.add(Convolution2D(32, 11,11, input_shape=(1, 25, 25), border_mode='valid',W_regularizer=regularizers.l2(0.01)))
dl.add(Activation('relu'))
dl.add(MaxPooling2D((2, 2)))

dl.add(Convolution2D(32, 3, 3, border_mode='valid', W_regularizer=regularizers.l2(0.01)))
dl.add(Activation('relu'))
dl.add(MaxPooling2D((3, 3)))

#*FILTER_SIZES[2]
dl.add(Convolution2D(32, 3, 3, border_mode='valid', W_regularizer=regularizers.l2(0.01)))
dl.add(Activation('relu'))
dl.add(MaxPooling2D((3, 3)))

dl.add(LRN2D())
dl.add(Flatten())

dl.add(Dropout(0.2))

dl.add(Dense(64))
dl.add(Activation('relu'))
dl.add(Dropout(0.1))


dl.add(Dense(1))
dl.add(Activation('sigmoid'))


print 'Compiling...'
# dl.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')


dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# X_train_image = np.load()



print 'Loading data...'
data = np.load('../../../jetimages.npy')
print 'there are {} training points'.format(data.shape[0])
print 'it is {}% signal'.format(data['signal'].mean() * 100)
print 'convering to images...'
X = data['image'].reshape((data.shape[0], 1, 25, 25)).astype('float32')
y = data['signal'].astype('float32')
print data['image'].shape

print 'extracting weights...'
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

signal = (signal == 1)
background = (signal == False)



# -- calculate the weights
weights = np.ones(data.shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

print 'calculating weights...'
weights[signal] = get_weights(reference_distribution, pt[signal], 
    bins=np.linspace(250, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
    bins=np.linspace(250, 300, 200))

num_inputs=2000000

idx = range(X.shape[0])
np.random.shuffle(idx)
X = X[idx][:val]
y = y[idx][:num_inputs]
weights = weights[idx].astype('float32')[:num_inputs]

tr = 8000

try:

    h = dl.fit(X[:tr], y[:tr], batch_size=3*32, nb_epoch=20, show_accuracy=True, 
	               validation_data = (X[tr:], y[tr:]), 
	               callbacks = 
	               [
	                   EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
	                   ModelCheckpoint('NewSLACNetConvNormalized-final-logloss.h5', monitor='val_loss', verbose=True, save_best_only=True),
	                   ROCModelCheckpoint('NewSLACNetConvNormalized-final-roc.h5', X[tr:], y[tr:], weights[tr:], verbose=True)
	               ])
except KeyboardInterrupt:
	print 'ended early!'


np.save("test.txt", yhat.astype('float32'))
log('Done')


#with open('./NewSLACNetConv-final.yaml', 'wb') as f:
#	f.write(dl.to_yaml())


