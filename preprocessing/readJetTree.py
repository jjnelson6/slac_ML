import ROOT as R
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math as m
R.gROOT.SetBatch(1)

ljmetDir = '/user_data/nelson/CMSSW_8_0_4/src/Delphes'
samplefile = '/delphes-tree-grids.root'

tFile = R.TFile.Open(ljmetDir+samplefile, "read")
tTree = tFile.Get("jetgrids")

count=0
entries=[]


def plot_mean_jet(rec, field = 'images', title = 'Average Jet Image'):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.mean(rec, axis = 0),  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    plt.title(r''+title)
    return fig


def plot_jet(rec, title = 'Jet Image', log=True):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    if log:
        im = ax.imshow(rec,  norm=LogNorm(vmin=0.00001, vmax=1), interpolation='nearest')
    else:
        im = ax.imshow(rec, interpolation='nearest')
    plt.title(r''+title)
    return fig




for event in tTree: 
      count+=1
      for ijet in range(len(event.JetAK8_ETA)):
            Jetarray=np.array(np.zeros((25,25)))
            etacenter=event.JetAK8_ETA[ijet]
            phicenter=event.JetAK8_PHI[ijet]
#            print 'jet#:',ijet,'etacenter:',event.JetAK8_ETA[ijet],'phicenter',event.JetAK8_PHI[ijet]
            for calo in range(len(event.CaloTower_ET)):
                       etadistance= int(abs(event.CaloTower_ETA[calo]-etacenter)/.0714)
                       phidistance= int(abs(event.CaloTower_PHI[calo]-phicenter)/(m.pi/44))
                       
                       if (etacenter > event.CaloTower_ETA[calo] and phicenter > event.CaloTower_PHI[calo]) and (etadistance<=12 and phidistance<=12):
                            Jetarray[12-etadistance][12-phidistance]=event.CaloTower_ET[calo]
                             #print etacenter, phicenter,event.CaloTower_ETA[calo], event.CaloTower_PHI[calo]
                             #print etadistance, phidistance
                            # continue
                       elif (etacenter > event.CaloTower_ETA[calo] and phicenter < event.CaloTower_PHI[calo]) and (etadistance<=12 and phidistance<=12):
                             Jetarray[12-etadistance][12+phidistance]=event.CaloTower_ET[calo]
                             
                       elif (etacenter < event.CaloTower_ETA[calo] and phicenter > event.CaloTower_PHI[calo]) and (etadistance<=12 and phidistance<=12):
                             Jetarray[12+etadistance][12-phidistance]=event.CaloTower_ET[calo]
                             #print etacenter, phicenter ,event.CaloTower_ETA[calo], event.CaloTower_PHI[calo]
                             #print etadistance, phidistance

                       elif (etacenter < event.CaloTower_ETA[calo] and phicenter < event.CaloTower_PHI[calo]) and (etadistance<=12 and phidistance<=12):
                             Jetarray[12+etadistance][12+phidistance]=event.CaloTower_ET[calo]
                             
            
            entries.append(Jetarray) 
            #print count
            
plot_mean_jet(entries).savefig('test_signal_avg.pdf')
#np.savez('Jet_test_plot',images=entries)

#                
                       #print 'LowerBinETA:', event.CaloEdgeEtaMin[calo], ' HigherBinEta:',event.CaloEdgeEtaMax[calo], ' LowerBinPhi:',event.CaloEdgePhiMin[calo], ' CaloET:',event.CaloTower_ET[calo]
            

