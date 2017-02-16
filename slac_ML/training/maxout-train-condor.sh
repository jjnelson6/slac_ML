#!/bin/bash

condorDir=$PWD
input=$1
outputDir=$2
runDir=$3

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $runDir
eval `scramv1 runtime -sh`

scl enable python27 bash

python run_maxout_train.py ${input}.npz $outputDir 
