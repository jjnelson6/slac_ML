#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

void AnalyseEvents(ExRootTreeReader *treeReader)
{
  TClonesArray *branchCaloTower = treeReader->UseBranch("CaloTower");
  //TClonesArray *branchEFlowTower = treeReader->UseBranch("EFlowTower");
  TClonesArray *branchJetAK8 = treeReader->UseBranch("JetAK8");

  Long64_t allEntries = treeReader->GetEntries();

  cout << "** Chain contains " << allEntries << " events" << endl;

  GenParticle *particle;
  //Electron *electron;
  //Photon *photon;
  //Muon *muon;

  //Track *track;
  Tower *tower;
 
  Jet *jet;
  TObject *object;

  TLorentzVector momentum;

  Float_t Eem, Ehad;
  Bool_t skip;

  Long64_t entry;

  Int_t i, j, k, pdgCode;
  Int_t etaBinEdge, etaBins = 31; //half of total number of bins
  Float_t etaBinStep = .0714;
  
  const Float_t pi = 3.14159;
  Int_t phiBinEdge, phiBins=44; //half of total number of  bins
  Float_t phiBinStep = pi/phiBins;
  Int_t count_jet=0;
  Int_t count_tower=0;


  for(entry = 0; entry < allEntries; ++entry)
    {
      // Load selected branches with data from specified event
      treeReader->ReadEntry(entry);
      for(i = 0; i < branchJetAK8->GetEntriesFast(); ++i)
	{
	  jet = (Jet*) branchJetAK8->At(i);

	  //	  momentum.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
	  if (jet->PT >= 200 and jet->PT <= 300)
	    {
	  

	   for(etaBinEdge = -etaBins+12 ; etaBinEdge <etaBins-12; ++etaBinEdge)
	     {
	       if ( (jet->Eta <= etaBinStep*(etaBinEdge+1)) and jet->Eta >= etaBinStep*etaBinEdge)
		 {
		   for (phiBinEdge = -phiBins; phiBinEdge < pi; ++phiBinEdge)
		     { 
		       if ((jet->Phi <= phiBinStep*(phiBinEdge+1)) and jet->Phi >= phiBinStep*phiBinEdge)
			 {
			   count_jet+=1;
			   for (k = 0 ; k <branchCaloTower->GetEntriesFast(); ++k)
			     {
			       tower= (Tower*) branchCaloTower->At(k);
			       if( (jet->Eta >= tower->Edges[0] and jet->Eta<= tower->Edges[1]) and (jet->Phi>=tower->Edges[2] and jet->Phi<=tower->Edges[3]))
				 {
				   count_tower+=1;
				   cout <<"CaloEdges: "<<tower->Edges[0]<<", "<<tower->Edges[2] <<endl;
				   cout <<"JetEta: "<<jet->Eta <<" JetPhi: "<<jet->Phi<<endl;
				    cout <<" CaloE: "<<tower->ET<<endl;
				 }
				   
			       // cout <<"tower: "<<branchCaloTower->GetEntriesFast()<<endl;
			     }//cout<<"JEt:"<<branchJetAK8->GetEntriesFast()<<" ,Cal: "<<branchCaloTower->GetEntriesFast()<<endl;
			   


			   //			   cout<<"Looping over jet constituents. Jet pt: "<<jet->PT<<", eta: "<<jet->Eta<<", phi: "<<jet->Phi<<endl;
			   //  cout<< " etabins: "<<etaBinStep*etaBinEdge<<", "<< etaBinStep*(etaBinEdge+1) <<" phibins: "<< phiBinStep*phiBinEdge <<", "<<phiBinStep*(phiBinEdge+1) <<endl;
			 }
		     }
		 }
	     }
	    }


	  /*
	
	  // Loop over all jet's constituents
	  for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j)
	    {
	      object = jet->Constituents.At(j);

	      // Check if the constituent is accessible
	      if(object == 0) continue;

	      if(object->IsA() == GenParticle::Class())
		{
		  particle = (GenParticle*) object;
		  // cout << "    GenPart pt: " << particle->PT << ", eta: " << particl\
		  particle->Eta << ", phi: " << particle->Phi << endl;
		  momentum += particle->P4();
		}
	      else if(object->IsA() == Tower::Class())
		{
      		  tower = (Tower*) object;
		  //		   cout << "    Tower pt: " << tower->ET << ", eta: " << tower->Eta <\< ", phi: " << tower->Phi << endl;
		  momentum += tower->P4();
		}
		}*/
	  //plots->fJetDeltaPT->Fill((jet->PT - momentum.Pt())/jet->PT);
	}
    }
  cout<<"Jet numbers:  "<<count_jet<<" Tower numbers: "<<count_tower<<endl;
}






void MakeJetGrids( const char *inputFile)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalyseEvents(treeReader);
}
