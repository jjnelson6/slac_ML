#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include <TROOT.h>
#include <TFile.h>
#include <math.h>
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#include "vector"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

void AnalyseEvents(ExRootTreeReader *treeReader, const char *outTree)
{
  TClonesArray *branchCaloTower = treeReader->UseBranch("CaloTower");
  //TClonesArray *branchEFlowTower = treeReader->UseBranch("EFlowTower");
  TClonesArray *branchJetAK8 = treeReader->UseBranch("JetAK8");

  Long64_t allEntries = treeReader->GetEntries();

  cout << "** Chain contains " << allEntries << " events" << endl;

  GenParticle *particle;
  //Electron *electron;
  //Photon *pho
  Tower *tower;

  Jet *jet;
  TObject *object;

  TLorentzVector subjet1;
  TLorentzVector subjet2;


  //  Float_t eem, ehad;
  //Bool_t skip;

  Long64_t entry;
  Int_t count_entries=0;
  Int_t i, j, k, pdgcode;
  Int_t etaBinEdge, etaBins = 34; //half of total number of bins
  Float_t etaBinStep = .0714;
  Float_t etaLowEdge = -2.4276;
  
  const Float_t pi = M_PI;
  Int_t phiBinEdge, phiBins=44; //half of total number of  bins
  Float_t phiBinStep = pi/phiBins;
  Float_t phiLowEdge = -1*pi;
  Int_t count_jet=0;
  Int_t count_tower=0;
  
  TTree *outputTree = new TTree("jetgrids","jetgrids");
  outputfile= new TFile(outTree,"recreate");

  float JetAK8_ETA;
  float JetAK8_PHI;
  float JetAK8_PT;
  float JetAK8_MASS;
  float JetAK8_Tau21;
  float JetAK8_Tau2;
  float JetAK8_Tau1;
  float JetAK8_LsubjetPt;
  float JetAK8_LsubjetEta;
  float JetAK8_LsubjetPhi;
  float JetAK8_SLsubjetPt;
  float JetAK8_SLsubjetEta;
  float JetAK8_SLsubjetPhi;
  float JetAK8_SubLeadingEta;
  float JetAK8_SubLeadingPhi;
  float JetAK8_PCEta;
  float JetAK8_PCPhi;
  vector<float> CaloTower_ET;
  vector<float> CaloEdgeEtaMin;
  vector<float> CaloEdgeEtaMax;
  vector<float> CaloEdgePhiMax;
  vector<float> CaloEdgePhiMin;
  vector<float> CaloTower_ETA;
  vector<float> CaloTower_PHI;
  
  outputTree->Branch("JetAK8_ETA",&JetAK8_ETA,"JetAK8_ETA/F");
  outputTree->Branch("JetAK8_PHI",&JetAK8_PHI,"JetAK8_PHI/F");
  outputTree->Branch("JetAK8_PT",&JetAK8_PT,"JetAK8_PT/F");
  outputTree->Branch("JetAK8_MASS",&JetAK8_MASS,"JetAK8_MASS/F");
  outputTree->Branch("JetAK8_Tau21",&JetAK8_Tau21,"JetAK8_Tau21/F");
  outputTree->Branch("JetAK8_Tau1",&JetAK8_Tau1,"JetAK8_Tau1/F");
  outputTree->Branch("JetAK8_Tau2",&JetAK8_Tau2,"JetAK8_Tau2/F");
  outputTree->Branch("JetAK8_LsubjetPt",&JetAK8_LsubjetPt,"JetAK8_LsubjetPt/F");
  outputTree->Branch("JetAK8_LsubjetEta",&JetAK8_LsubjetEta,"JetAK8_LsubjetEta/F");
  outputTree->Branch("JetAK8_LsubjetPhi",&JetAK8_LsubjetPhi,"JetAK8_LsubjetPhi/F");
  outputTree->Branch("JetAK8_SLsubjetPt",&JetAK8_SLsubjetPt,"JetAK8_SLsubjetPt/F");
  outputTree->Branch("JetAK8_SLsubjetEta",&JetAK8_SLsubjetEta,"JetAK8_SLsubjetEta/F");
  outputTree->Branch("JetAK8_SLsubjetPhi",&JetAK8_SLsubjetPhi,"JetAK8_SLsubjetPhi/F");
  outputTree->Branch("JetAK8_SubLeadingEta",&JetAK8_SubLeadingEta,"JetAK8_SubLeadingEta/F");
  outputTree->Branch("JetAK8_SubLeadingPhi",&JetAK8_SubLeadingPhi,"JetAK8_SubLeadingPhi/F");
  outputTree->Branch("JetAK8_PCEta",&JetAK8_PCEta,"JetAK8_PCEta/F");
  outputTree->Branch("JetAK8_PCPhi",&JetAK8_PCPhi,"JetAK8_PCPhi/F");
  outputTree->Branch("CaloEdgeEtaMin",&CaloEdgeEtaMin);
  outputTree->Branch("CaloEdgeEtaMax",&CaloEdgeEtaMax);
  outputTree->Branch("CaloEdgePhiMin",&CaloEdgePhiMin);
  outputTree->Branch("CaloEdgePhiMax",&CaloEdgePhiMax);
  outputTree->Branch("CaloTower_ET",&CaloTower_ET);
  outputTree->Branch("CaloTower_ETA",&CaloTower_ETA);
  outputTree->Branch("CaloTower_PHI",&CaloTower_PHI);
  
  for(entry = 0; entry < allEntries; ++entry){
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    CaloTower_ET.clear();
    CaloEdgeEtaMin.clear();
    CaloEdgeEtaMax.clear();
    CaloEdgePhiMin.clear();
    CaloEdgePhiMax.clear();
    CaloTower_ETA.clear();
    CaloTower_PHI.clear();

    count_entries+=1;

    double maxpt = 0;    
    int maxptindex = -1;
    for(i = 0; i < branchJetAK8->GetEntriesFast(); ++i){
      jet = (Jet*) branchJetAK8->At(i);
     
      //lmomentum.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
      //smomentum.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
      //Selecting Jet PT
      if (jet->PT < 200 or jet->PT > 300) continue;
      if (jet->Mass < 65 or jet->Mass > 95) continue;

      if (jet->PT > maxpt){
	maxpt = jet->PT;
	maxptindex = i;
      }
    }

    if(maxptindex == -1) continue; // no jet matched the requirements

    jet = (Jet*) branchJetAK8->At(maxptindex); // get the highest pT jet in the windows
	
    // save subjets
    JetAK8_LsubjetPt = -99;
    JetAK8_LsubjetEta = -99;
    JetAK8_LsubjetPhi = -99;
    JetAK8_SLsubjetPt = -99;
    JetAK8_SLsubjetEta = -99;
    JetAK8_SLsubjetPhi = -99;
    JetAK8_SubLeadingEta = -99;
    JetAK8_SubLeadingPhi = -99;
    JetAK8_PCEta = -99;
    JetAK8_PCPhi = -99;

    if(jet->NSubJetsSoftDropped > 0){
      subjet1 = jet->SoftDroppedP4[1];
      JetAK8_LsubjetPt = subjet1.Pt();
      JetAK8_LsubjetEta = subjet1.Eta();
      JetAK8_LsubjetPhi = subjet1.Phi();
    }

    if(jet->NSubJetsSoftDropped > 1) {
      
      subjet2 = jet->SoftDroppedP4[2];  
      JetAK8_SLsubjetPt = subjet2.Pt();
      JetAK8_SLsubjetEta = subjet2.Eta();
      JetAK8_SLsubjetPhi = subjet2.Phi();
      
      JetAK8_SubLeadingEta = subjet2.Eta() - subjet1.Eta();
      JetAK8_SubLeadingPhi = subjet2.DeltaPhi(subjet1);
  
    }
    
      //if(subjet1.Pt() < subjet2.Pt()) cout << "subjets might not be pt ordered" << endl;
     


    // save tau
    JetAK8_Tau1 = jet->Tau[0];
    JetAK8_Tau2 = jet->Tau[1];
    JetAK8_Tau21 = JetAK8_Tau2/JetAK8_Tau1;

    // Find jet center
    double etacenter = jet->Eta;
    double phicenter = jet->Phi;
    	
    


    count_jet+=1;
    //cout<< "Jet#:"<<i<<" etacenter:"<< etacenter<<" phicenter:"<< phicenter<< " #"<<branchJetAK8->GetEntriesFast()<<endl;
    cout<< "Jet#:"<<i<<" etacenter:"<< etacenter<<" phicenter:"<< phicenter<<" JetPT:" <<jet->PT  << " #"<<branchJetAK8->GetEntriesFast()<<endl;

    
    JetAK8_ETA = etacenter;
    JetAK8_PHI = phicenter;
    JetAK8_PT = jet->PT;
    JetAK8_MASS = jet->Mass;
    
    // Find distance from lowest eta.
    double etaToLowEdge = etacenter - etaLowEdge;
    double phiToLowEdge = phicenter;  // pi/2 - (-pi) = 3pi/2;  -pi/4 - (-pi) = 3pi/4;  
    
    // Find the bin indices -- putting this as "int" should truncate/round to the closest
    int etaIndex = etaToLowEdge/etaBinStep;
    int phiIndex = phiToLowEdge/phiBinStep;
    
    // For phi we can wrap, so find the right high and low phi values
    
    // For simplicity, let's always use the same corner of the bin (the bottom left corner)
    // N = 44, grid runs -44 to 44. Up = 13 above center. Down = 12 below center
    int phiIndexDn = phiIndex-12;       // downpoint = C-D 
    int phiIndexUp = phiIndex+13;       // uppoint = C+U
    if(phiIndex < -32){                 // if C < -N + D --> C < -44 + 12 --> C < -32
      phiIndexDn = 88 + (phiIndex-12);  // downpoint = -2N + (C-D)
      phiIndexUp = (phiIndex+13);       // uppoint = C+U
    }else if(phiIndex > 31){            // if C > N - U --> C > 44 - 13 --> C > 31
      phiIndexDn = (phiIndex-12);       // downpoint = C-D 
      phiIndexUp = -88 + (phiIndex+13);    // uppoint = -2N + (C+U)
    }
    double phiLowEdge = phiIndexDn*phiBinStep;
    double phiHighEdge = phiIndexUp*phiBinStep;
    //
    //count_tower=0;
    // Loop over towers to find the towers in this region
    for(k = 0; k < branchCaloTower->GetEntriesFast(); ++k){
      tower = (Tower*) branchCaloTower->At(k);
      
      // skip towers that are lower than low eta or higher than high eta
      if(tower->Edges[0] <= (etaLowEdge + (etaIndex-12)*etaBinStep) || tower->Edges[0] >= (etaLowEdge + (etaIndex+13)*etaBinStep)) continue;
      
      // skip towers above/below the phi band when it does not wrap
      if(phiHighEdge > phiLowEdge && (tower->Edges[2] < phiLowEdge || tower->Edges[2] > phiHighEdge)) continue;
      // skip towers that are in between low phi and high phi when it wraps
      if(phiHighEdge < phiLowEdge && (tower->Edges[2] < phiLowEdge && tower->Edges[2] > phiHighEdge)) continue;

      count_tower+=1;
      // cout<<"PhiLow:"<<phiLowEdge <<" Phihigh:"<<phiHighEdge<<" toweredge:"<<tower->Edges[2]<<endl;
      CaloTower_ET.push_back(tower->ET);
      CaloEdgeEtaMin.push_back(tower->Edges[0]);
      CaloEdgeEtaMax.push_back(tower->Edges[1]);
      CaloEdgePhiMin.push_back(tower->Edges[2]);
      CaloEdgePhiMax.push_back(tower->Edges[3]);
      CaloTower_PHI.push_back(tower->Phi);
      CaloTower_ETA.push_back(tower->Eta);
      
    }


    // Get constituents (following example3.C...)
    TObject *object;
    GenParticle *particle;
    Track *track;
    Tower *tower;
    vector<pair<double,double>> consts_image;
    vector<double> consts_E;
    int nconsts = 0;
    for(int iconst = 0; iconst < jet->Constituents.GetEntriesFast(); iconst++){
      object = jet->Constituents.At(iconst);
      if(object == 0) continue;
      
      nconsts++;
      pair<double,double> const_hold;
      TLorentzVector constituent;

      if(object->IsA() == GenParticle::Class()){
	particle = (GenParticle*)object;

	constituent = particle->P4();
	const_hold.first = particle->Eta - subjet1.Eta();
	const_hold.second = constituent.DeltaPhi(subjet1);
	consts_image.push_back(const_hold);
	consts_E.push_back(constituent.E());
	
      }else if(object->IsA() == Track::Class()){
	track = (Track*)object;

	constituent = track->P4(); 
	const_hold.first = track->Eta - subjet1.Eta();
	const_hold.second = constituent.DeltaPhi(subjet1);
	consts_image.push_back(const_hold);
	consts_E.push_back(constituent.E());
	
      }else if(object->IsA() == Tower::Class()){
	tower = (Tower*)object;

	constituent = tower->P4();
	const_hold.first = tower->Eta - subjet1.Eta();
	const_hold.second = constituent.DeltaPhi(subjet1);
	consts_image.push_back(const_hold);
	consts_E.push_back(constituent.E());
	
      }
    }

    //Quickly run PCA for the rotation.
    double xbar = 0.;
    double ybar = 0.;
    double x2bar = 0.;
    double y2bar = 0.;
    double xybar = 0.;
    double n = 0;
    
    for(int iconst = 0; iconst < nconsts; iconst++)
      {
        double x = consts_image[iconst].first;
        double y = consts_image[iconst].second;
	double E = consts_E[iconst];
        n+=E;
        xbar+=x*E;
        ybar+=y*E;
      }

    double mux = xbar / n;
    double muy = ybar / n;

    xbar = 0.;
    ybar = 0.;
    n = 0.;

    for(int iconst = 0; iconst < nconsts; iconst++)
      {
        double x = consts_image[i].first - mux;
        double y = consts_image[i].second - muy;
	double E = consts_E[iconst];
        n+=E;
        xbar+=x*E;
        ybar+=y*E;
        x2bar+=x*x*E;
        y2bar+=y*y*E;
        xybar+=x*y*E;
      }

    double sigmax2 = x2bar / n - mux*mux;
    double sigmay2 = y2bar / n - muy*muy;
    double sigmaxy = xybar / n - mux*muy;
    double lamb_min = 0.5* ( sigmax2 + sigmay2 - sqrt( (sigmax2-sigmay2)*(sigmax2-sigmay2) + 4*sigmaxy*sigmaxy) );
    double lamb_max = 0.5* ( sigmax2 + sigmay2 + sqrt( (sigmax2-sigmay2)*(sigmax2-sigmay2) + 4*sigmaxy*sigmaxy) );

    double dir_x = sigmax2+sigmaxy-lamb_min;
    double dir_y = sigmay2+sigmaxy-lamb_min;
    
    //The first PC is only defined up to a sign.  Let's have it point toward the side of the jet with the most energy.

    double Eup = 0.;
    double Edn = 0.;
    
    for(int iconst = 0; iconst < nconsts; iconst++)
      {
	double x = consts_image[i].first - mux;
        double y = consts_image[i].second - muy;
	double E = consts_E[iconst];
	double dotprod = dir_x*x+dir_y*y;
	if (dotprod > 0) Eup+=E;
	else Edn+=E;
      }
    
    if (Edn < Eup){
      dir_x = -dir_x;
      dir_y = -dir_y;
    }

    JetAK8_PCEta = dir_x;
    JetAK8_PCPhi = dir_y;

    outputTree->Fill(); 
  }

  cout<<"Jet numbers:  "<<count_jet<<" Tower numbers: "<<count_tower<<endl;
  cout<<"entries: "<<count_entries<<endl;
  outputTree->Write();
}


void MakeJetGridsTree( const char *inputFile,const char *outputFile)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalyseEvents(treeReader,outputFile);
}
