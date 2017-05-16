Creating jet images with pileup and using MaxOut Deep Neural Network

These instructions are for running setup on Brux6. This is because of limitations of certain environments used on this cluster, thus some steps may not be necessary onother systems.

Software used: Python 2.7.11 and 3.4.2
Packages: Keras 2.0.3, Theano 0.9.0, Scikit-Learn 0.18.1, and Scikit-Image 0.13.0  

Example Steps for Making Jet Images


1. Build  CMMSSW work area: 

cmsrel CMSSW_8_0_4 
cd CMSSW_8_0_4/src

2. Build Delphes Environment(https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/ReadingCMSFiles): 

export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.4.1.tar.gz
tar -zxf Delphes-3.4.1.tar.gz

cd Delphes-3.4.1
make -j 4

3. Check Delphes card

The card can be found in slac_ML/preprocessing/Delphes_Jetimage.tcl. The size of the Calorimeter can be adjusted in 'module Calorimeter Calorimeter'. Currently it is set to have 44 phi bins and 32 eta bins. The step size in phi is pi/44 and the step size in eta is 0.0714. This is based off of the example card CMSSW_8_0_4/src/Delphes/cards/delphes_card_CMS_PileUp.tcl 

4. Make Delphes Trees

5. Copy macro and python scripts:

cp slac_ML/preprocessing/readJetTree.py /CMSSW_8_0_4/src/Delphes-3.4.1/ 

cp slac_ML/preprocessing/MakeJetGridsTree.C /CMSSW_8_0_4/src/Delphes-3.4.1/

cp slac_ML/preprocessing/makeJetGrids.sh /CMSSW_8_0_4/src/Delphes-3.4.1/

cp slac_ML/preprocessing/makeJetGridscondor.py /CMSSW_8_0_4/src/Delphes-3.4.1/

6. Extract Delphes tree info and make jet grids: 

cmsenv 

(Make some changes to file directories in makeJetGridscondor.py. Currently saves to hadoop. Check shell script for details how this is done)

python makeJetGridscondor.py <file with Delphes trees>

(Then make edits to file directories to readJetTree.py.) 

python readJetTree.py 

(Open another shell in Brux6 and go to preprocessing directory.)

To initiate proper environment: scl enable rh-python34 bash

python rotateJetGrid.py <filename>

7.Plot Images

(make changes to plot directories for plotjetimages.py. This will make heat maps of the unrotated and rotated images.)

python plotjetimages.py <filename> 



Running MaxOut Deep Neural Network

1. To initiate proper environment: scl enable rh-python34 bash 

2.Open slac_ML/training/maxout-train.py. Insert data file. Use num_inputs to set how many samples you want to used for testing and training. Use Kfolds to set how many samples you want for testing and training. In the fit function set filenames for .h5 which save the weights the network learns. 

3. Use slac_ML/training/maxout-test.py. This requires datafile to be tested, weights, if you want to normalize, and outputfile name. This will create a numpy array with the output values from the network. 

4. Use Distbution_plotter.py, TestingROC_plottter.py, makehists, and/or MultiROC_plottter.py to make plots that evaluate the performace of the network.   

 