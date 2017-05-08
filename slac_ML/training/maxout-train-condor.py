import logging, os, sys, datetime, time
import numpy as np

#######Use this for serializing the cross-validation training process for MaxOut#######

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

cTime=datetime.datetime.now()
date='%i_%i_%i_%i_%i_%i'%(cTime.year,cTime.month,cTime.day,cTime.hour,cTime.minute,cTime.second)

foldN=0
for nfile in npzfiles:
    rawname= nfile[:-5] 
    foldN+=1
    dict={'RUNDIR':runDir, 'CONDORDIR':condorDir, 'OUTPUTDIR':outputDir, 'FILENAME':rawname}
    jdfName=condorDir+'/%(FILENAME)s.job'%dict
    print (jdfName)
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
    print (foldN, "jobs submitted!!!")



print("--- %s minutes ---" % (round(time.time() - start_time, 2)/60))

