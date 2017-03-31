'''
maxout-train.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to train a maxout net

'''
import logging, os, sys, datetime, time
#from keras.models import Sequential, model_from_yaml
#from keras.layers.core import *
#from keras.layers import containers
#from keras.optimizers import *
#from keras import regularizers
#from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
#from sklearn.metrics import roc_curve, auc

#from viz import *

start_time= time.time()
shift= sys.argv[1]
inputDir= '/user_data/nelson/slac_ML/slac_ML/training/'+shift
outputDir= '/user_data/nelson/training_output/'+shift
runDir= os.getcwd()

condorDir= outputDir+'/condorLogs/'

npzfiles= os.popen('ls '+inputDir)
os.system('mkdir -p '+outputDir)
os.system('mkdir -p '+condorDir)


LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)

# X_train_image = np.load()

cTime=datetime.datetime.now()
date='%i_%i_%i_%i_%i_%i'%(cTime.year,cTime.month,cTime.day,cTime.hour,cTime.minute,cTime.second)


#KFold(X.shape[0],10)
foldN=0
for nfile in npzfiles:
    #for train_ix, test_ix in kf:
    rawname= nfile[:-5] 
    foldN+=1
    dict={'RUNDIR':runDir, 'CONDORDIR':condorDir, 'OUTPUTDIR':outputDir, 'FILENAME':rawname}
    jdfName=condorDir+'/%(FILENAME)s.job'%dict
    print jdfName
    jdf=open(jdfName,'w')
    jdf.write(
        
"""universe = vanilla
Executable = %(RUNDIR)s/maxout-train-condor.sh 
Should_Transfer_Files = NO
Transfer_Input_Files = 
Output = %(FILENAME)s.out
Error = %(FILENAME)s.err
Log = %(FILENAME)s.log
Notification = Never

Arguments = %(FILENAME)s %(OUTPUTDIR)s %(RUNDIR)s
Queue 1""" %dict)
    jdf.close()
    os.chdir('%s/'%(condorDir))
    os.system('condor_submit %(FILENAME)s.job'%dict)
    os.system('sleep 0.5')                                
    os.chdir('%s'%(runDir))
    print foldN, "jobs submitted!!!"



print("--- %s minutes ---" % (round(time.time() - start_time, 2)/60))

