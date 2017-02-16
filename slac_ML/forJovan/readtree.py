import pickle
import ROOT as R

R.gROOT.SetBatch(1)

ljmetDir = '/user_data/nelson/2016analysis/CMSSW_7_4_14/src/2016_step3/LJMET_MuonId_Toptags/nominal/'
samplefile = 'TprimeTprime_M-900_TuneCUETP8M1_13TeV-madgraph-pythia8_TZTZ_hadd.root'

#'ZZ_TuneCUETP8M1_13TeV-pythia8_hadd.root'
#

tFile = R.TFile.Open(ljmetDir+samplefile, "read")
tTree = tFile.Get("ljmet")

myCollection = {}
myCollection['inputs'] = []
myCollection['Ttagg']=[]



for event in tTree: 


      #collectionID = str(event.event_CommonCalc)+'_'+str(event.run_CommonCalc)+'_'+str(event.lumi_CommonCalc)
      
      for ijet in range(len(event.theJetAK8Pt_JetSubCalc_PtOrdered)):
            istagged=event.theJetAK8Tmatch_JetSubCalc_PtOrdered[ijet]
            pt= event.theJetAK8Pt_JetSubCalc_PtOrdered[ijet]
            eta=event.theJetAK8Eta_JetSubCalc_PtOrdered[ijet]
            SoftDropMass=event.theJetAK8SoftDropMass_JetSubCalc_PtOrdered[ijet]
            Tau32=event.theJetAK8NjettinessTau3_JetSubCalc_PtOrdered[ijet]/event.theJetAK8NjettinessTau2_JetSubCalc_PtOrdered[ijet]
      
            print 'event:',event.event_CommonCalc,'jet#:',ijet,'AK8_pT:',event.theJetAK8Pt_JetSubCalc_PtOrdered[ijet]
            myCollection['inputs'].append([pt,eta,SoftDropMass,Tau32])
            myCollection['Ttagg'].append(istagged)

print len(myCollection['Ttagg'])      
pickle.dump(myCollection,open('myCollection.p','wb'))
