#!/bin/bash

infilename=${1}
outfilename=${2}
inputDir=${3}
outputDir=${4}
runDir=${5}

scratch=${PWD}
macroDir=${PWD}

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc491

#scramv1 project CMSSW CMSSW_8_0_4
#cd CMSSW_8_0_4

cd $runDir 
eval `scramv1 runtime -sh`
#export PATH=$PATH:$runDir

root -l -b -q MakeJetGridsTree_Julie.C\(\"$inputDir/$infilename\",\"$outfilename\"\)

cp $outfilename $outputDir
rm $outfilename

echo "Done"


