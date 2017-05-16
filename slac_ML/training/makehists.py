import numpy as np
from ROOT import *
import matplotlib.pyplot as plt

network=np.load('../../../slac_ML/slac_ML/training/train35PU_test35PU.npy')
data=np.load('../../../Jetinfo35PU_noimage.npz')
output='train35PU_test35PU'


mass=data['jet_mass']
tau21=data['tau21']
net=network

c1=TCanvas("c1","c1",800,600)
h=TH2D("2dplot","DNN vs Mass",100,65,95,100,0,1)
h.GetXaxis().SetTitle("Jet Mass (GeV)")
h.GetYaxis().SetTitle("DNN Output")
for k in range(len(net)): 
#    print k
    
    h.Fill(mass[k],net[k])



gPad.SetLogz(1)
gStyle.SetOptStat(0)
#c1.SetXTitle("Jet Mass (GeV)")
h.Draw("colz")
c1.SaveAs(output+"_DNNvsmass.pdf")

c2=TCanvas("c2","c2",800,600)
w=TH2D("2dplot1","DNN vs Tau21",100,0,1,100,0,1)
w.GetXaxis().SetTitle("Tau21")
w.GetYaxis().SetTitle("DNN Output")
for k in range(len(net)): 
#    print k
    
    w.Fill(tau21[k],net[k])


gPad.SetLogz(1)
gStyle.SetOptStat(0)
#gStyle.SetXaxisTitle("Tau21")
w.Draw("colz")
c2.SaveAs(output+"_DNNvstau21.pdf")







"""
#print '{} jets before preselection'.format(data.shape[0])
data = np.load('../../jetimages.npy')
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']
signal = (signal == 1)
background = (signal == 0)
plt.figure()
plt.hist(tau_21[background],bins=100)
plt.savefig('QCDTau21_SLAC.pdf')


data = np.load('../../preprocessing/Jetimages_0PU.npz')
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau21']
signal = (signal == 1)
background = (signal == 0)
plt.figure()
plt.hist(tau_21[background],bins=100)
plt.savefig('QCDTau21_0PU.pdf')
"""
