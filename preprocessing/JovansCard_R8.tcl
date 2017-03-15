#
#  Phase II - No Pile-Up
#
#  Main authors: Michele Selvaggi (UCL)
#
#  Released on:
#
#  Version: v01
#
#
#######################################
# Order of execution of various modules
#######################################

set ExecutionPath {

  PileUpMerger
  ParticlePropagator

  ChargedHadronTrackingEfficiency
  ElectronTrackingEfficiency
  MuonTrackingEfficiency

  ChargedHadronMomentumSmearing
  ElectronEnergySmearing
  MuonMomentumSmearing

  TrackMerger
  Calorimeter
  ElectronFilter
  TrackPileUpSubtractor
  RecoPuFilter
  NeutralEFlowMerger
  EFlowMergerCHS
  Rho

  NeutrinoFilter

  GenJetFinderAK8
  FastJetFinderAK8
  JetPileUpSubtractorAK8
  JetEnergyScaleAK8
  JetFlavorAssociationAK8

  GenParticleFilter



  TreeWriter
}

###############
# PileUp Merger
###############

module PileUpMerger PileUpMerger {
  set InputArray Delphes/stableParticles

  set ParticleOutputArray stableParticles
  set VertexOutputArray vertices

  # pre-generated minbias input file
  #set PileUpFile ../eos/cms/store/group/upgrade/delphes/PhaseII/MinBias_100k.pileup
  set PileUpFile MinBias_100k.pileup

  # average expected pile up
  set MeanPileUp 0

  # maximum spread in the beam direction in m
  set ZVertexSpread 0.25

  # maximum spread in time in s
  set TVertexSpread 800E-12

  # vertex smearing formula f(z,t) (z,t need to be respectively given in m,s) - {exp(-(t^2/160e-12^2/2))*exp(-(z^2/0.053^2/2))}
  set VertexDistributionFormula {exp(-(t^2/160e-12^2/2))*exp(-(z^2/0.053^2/2))}

}


#################################
# Propagate particles in cylinder
#################################

module ParticlePropagator ParticlePropagator {
  set InputArray PileUpMerger/stableParticles
  #set InputArray Delphes/stableParticles

  set OutputArray stableParticles
  set ChargedHadronOutputArray chargedHadrons
  set ElectronOutputArray electrons
  set MuonOutputArray muons

  # radius of the magnetic field coverage, in m
  set Radius 1.29
  # half-length of the magnetic field coverage, in m
  set HalfLength 3.0

  # magnetic field
  set Bz 3.8
}


####################################
# Charged hadron tracking efficiency
####################################

module Efficiency ChargedHadronTrackingEfficiency {
  ## particles after propagation
  set InputArray  ParticlePropagator/chargedHadrons
  set OutputArray chargedHadrons
  # tracking efficiency formula for charged hadrons
  set EfficiencyFormula {
      (pt <= 0.2) * (0.00) + \
	  (abs(eta) <= 1.2) * (pt > 0.2 && pt <= 1.0) * (pt * 0.96) + \
	  (abs(eta) <= 1.2) * (pt > 1.0) * (0.97) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.5) * (pt > 0.2 && pt <= 1.0) * (pt*0.85) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.5) * (pt > 1.0) * (0.87) + \
	  (abs(eta) > 2.5 && abs(eta) <= 4.0) * (pt > 0.2 && pt <= 1.0) * (pt*0.8) + \
	  (abs(eta) > 2.5 && abs(eta) <= 4.0) * (pt > 1.0) * (0.82) + \
	  (abs(eta) > 4.0) * (0.00)
  }
}


#####################################
# Electron tracking efficiency - ID
####################################

module Efficiency ElectronTrackingEfficiency {
  set InputArray  ParticlePropagator/electrons
  set OutputArray electrons
  # tracking efficiency formula for electrons
  set EfficiencyFormula {
      (pt <= 0.2) * (0.00) + \
	  (abs(eta) <= 1.2) * (pt > 0.2 && pt <= 1.0) * (pt * 0.96) + \
	  (abs(eta) <= 1.2) * (pt > 1.0) * (0.97) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.5) * (pt > 0.2 && pt <= 1.0) * (pt*0.85) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.5) * (pt > 1.0 && pt <= 10.0) * (0.82+pt*0.01) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.5) * (pt > 10.0) * (0.90) + \
	  (abs(eta) > 2.5 && abs(eta) <= 4.0) * (pt > 0.2 && pt <= 1.0) * (pt*0.8) + \
	  (abs(eta) > 2.5 && abs(eta) <= 4.0) * (pt > 1.0 && pt <= 10.0) * (0.8+pt*0.01) + \
	  (abs(eta) > 2.5 && abs(eta) <= 4.0) * (pt > 10.0) * (0.85) + \
	  (abs(eta) > 4.0) * (0.00)

  }
}

##########################
# Muon tracking efficiency
##########################

module Efficiency MuonTrackingEfficiency {
  set InputArray ParticlePropagator/muons
  set OutputArray muons
  # tracking efficiency formula for muons
  set EfficiencyFormula {
      (pt <= 0.2) * (0.00) + \
	  (abs(eta) <= 1.2) * (pt > 0.2 && pt <= 1.0) * (pt * 1.00) + \
	  (abs(eta) <= 1.2) * (pt > 1.0) * (1.00) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.8) * (pt > 0.2 && pt <= 1.0) * (pt*1.00) + \
	  (abs(eta) > 1.2 && abs(eta) <= 2.8) * (pt > 1.0) * (1.00) + \
	  (abs(eta) > 2.8 && abs(eta) <= 4.0) * (pt > 0.2 && pt <= 1.0) * (pt*0.95) + \
	  (abs(eta) > 2.8 && abs(eta) <= 4.0) * (pt > 1.0) * (0.95) + \
	  (abs(eta) > 4.0) * (0.00)

  }
}


########################################
# Momentum resolution for charged tracks
########################################

module MomentumSmearing ChargedHadronMomentumSmearing {
  ## hadrons after having applied the tracking efficiency
  set InputArray  ChargedHadronTrackingEfficiency/chargedHadrons
  set OutputArray chargedHadrons
  # resolution formula for charged hadrons ,

  source trackMomentumResolution_vs_p_PIX4022.tcl
}

#################################
# Energy resolution for electrons
#################################

module EnergySmearing ElectronEnergySmearing {
  set InputArray ElectronTrackingEfficiency/electrons
  set OutputArray electrons

  # set ResolutionFormula {resolution formula as a function of eta and energy}

  # resolution formula for electrons

  # taking something flat in energy for now, ECAL will take over at high energy anyway.
  # inferred from hep-ex/1306.2016 and 1502.02701
  set ResolutionFormula {

                        (abs(eta) <= 1.5)  * (energy*0.028) +
    (abs(eta) > 1.5  && abs(eta) <= 1.75)  * (energy*0.037) +
    (abs(eta) > 1.75  && abs(eta) <= 2.15) * (energy*0.038) +
    (abs(eta) > 2.15  && abs(eta) <= 3.00) * (energy*0.044) +
    (abs(eta) > 3.00  && abs(eta) <= 4.00) * (energy*0.10)}

}

###############################
# Momentum resolution for muons
###############################

module MomentumSmearing MuonMomentumSmearing {
  set InputArray MuonTrackingEfficiency/muons
  set OutputArray muons
  # resolution formula for muons

  # up to |eta| < 2.8 take measurement from tracking + muon chambers
  # for |eta| > 2.8 and pT < 5.0 take measurement from tracking alone taken from
  # http://mersi.web.cern.ch/mersi/layouts/.private/Baseline_tilted_200_Pixel_1_1_1/index.html
  source muonMomentumResolution.tcl
}

##############
# Track merger
##############

module Merger TrackMerger {
# add InputArray InputArray
  add InputArray ChargedHadronMomentumSmearing/chargedHadrons
  add InputArray ElectronEnergySmearing/electrons
  add InputArray MuonMomentumSmearing/muons
  set OutputArray tracks
}

#############
# Calorimeter
#############

module Calorimeter Calorimeter {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray TrackMerger/tracks

  set TowerOutputArray towers
  set PhotonOutputArray photons
#  set EdgesOutputArray edges  
  set EFlowTrackOutputArray eflowTracks
  set EFlowPhotonOutputArray eflowPhotons
  set EFlowNeutralHadronOutputArray eflowNeutralHadrons

  set ECalEnergyMin 0.5
  set HCalEnergyMin 1.0

  set ECalEnergySignificanceMin 1.0
  set HCalEnergySignificanceMin 1.0

  set SmearTowerCenter true

    set pi [expr {acos(-1)}]

  # lists of the edges of each tower in eta and phi
  # each list starts with the lower edge of the first tower
  # the list ends with the higher edged of the last tower

  # 25X25 Jet-image bins


 #   set Phibins {}
 #   for {set i-44}{$i <=44}{incr i}{
  #      add Phibins[expr{$i *$pi/44}]
  #  }
  #  foreach eta {-0.8568 -0.7854 -0.7140 -0.6426 -0.5712 -0.4998 -0.4284 -0.3570 -0.2856 -0.2142 -0.1428 -0.0714 0.0 0.0714 0.1428 0.2142 0.2856 0.3570 0.4284 0.4998 0.5712 0.6426 0.7140 0.7854 0.8568}{
 #       add etaPhibins $eta $Phibins
 #   }


  set PhiBins {}
  for {set i -44} {$i <= 44} {incr i} {
    add PhiBins [expr {$i * $pi/44.0}]
  }
 # foreach eta {-0.9996 -0.9282 -0.8568 -0.7854 -0.7140 -0.6426 -0.5712 -0.4998 -0.4284 -0.3570 -0.2856 -0.2142 -0.1428 -0.0714 0.0 0.0714 0.1428 0.2142 0.2856 0.3570 0.4284 0.4998 0.5712 0.6426 0.7140 0.7854 0.8568 0.9282} {
#    add EtaPhiBins $eta $PhiBins
#  }

    for {set i -28} {$i <= 29} {incr i} {
    set eta [expr {$i * 0.0714}]
    add EtaPhiBins $eta $PhiBins
  }



    # default energy fractions {abs(PDG code)} {Fecal Fhcal}
    add EnergyFraction {0} {0.0 1.0}
  # energy fractions for e, gamma and pi0
    add EnergyFraction {11} {1.0 0.0}
    add EnergyFraction {22} {1.0 0.0}
    add EnergyFraction {111} {1.0 0.0}
  # energy fractions for muon, neutrinos and neutralinos
    add EnergyFraction {12} {0.0 0.0}
    add EnergyFraction {13} {0.0 0.0}
    add EnergyFraction {14} {0.0 0.0}
    add EnergyFraction {16} {0.0 0.0}
    add EnergyFraction {1000022} {0.0 0.0}
    add EnergyFraction {1000023} {0.0 0.0}
    add EnergyFraction {1000025} {0.0 0.0}
    add EnergyFraction {1000035} {0.0 0.0}
    add EnergyFraction {1000045} {0.0 0.0}
  # energy fractions for K0short and Lambda
    add EnergyFraction {310} {0.3 0.7}
    add EnergyFraction {3122} {0.3 0.7}

    # set ECalResolutionFormula {resolution formula as a function of eta and energy}
    set ECalResolutionFormula {                  (abs(eta) <= 3.0) * sqrt(energy^2*0.007^2 + energy*0.07^2 + 0.35^2) +
	(abs(eta) > 3.0 && abs(eta) <= 5.0) * sqrt(energy^2*0.107^2 + energy*2.08^2)}

    # set HCalResolutionFormula {resolution formula as a function of eta and energy}
    set HCalResolutionFormula {                  (abs(eta) <= 3.0) * sqrt(energy^2*0.050^2 + energy*1.50^2) +
	(abs(eta) > 3.0 && abs(eta) <= 5.0) * sqrt(energy^2*0.130^2 + energy*2.70^2)}
}


#################
# Electron filter
#################

module PdgCodeFilter ElectronFilter {
  set InputArray Calorimeter/eflowTracks
  set OutputArray electrons
  set Invert true
  add PdgCode {11}
  add PdgCode {-11}
}


##########################
# Track pile-up subtractor
##########################

module TrackPileUpSubtractor TrackPileUpSubtractor {
# add InputArray InputArray OutputArray
  add InputArray Calorimeter/eflowTracks eflowTracks
  add InputArray ElectronFilter/electrons electrons
  add InputArray MuonMomentumSmearing/muons muons

  set VertexInputArray PileUpMerger/vertices
  # assume perfect pile-up subtraction for tracks with |z| > fZVertexResolution
  # Z vertex resolution in m
  set ZVertexResolution 0.0001
}

########################
# Reco PU filter
########################

module RecoPuFilter RecoPuFilter {
  set InputArray Calorimeter/eflowTracks
  set OutputArray eflowTracks
}

####################
# Neutral eflow erger
####################

module Merger NeutralEFlowMerger {
# add InputArray InputArray
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflowTowers
}

############################
# Energy flow merger no PU
############################

module Merger EFlowMergerCHS {
# add InputArray InputArray
  add InputArray RecoPuFilter/eflowTracks
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflow
}

#################
# Neutrino Filter
#################

module PdgCodeFilter NeutrinoFilter {

  set InputArray Delphes/stableParticles
  set OutputArray filteredParticles

  set PTMin 0.0

  add PdgCode {12}
  add PdgCode {14}
  add PdgCode {16}
  add PdgCode {-12}
  add PdgCode {-14}
  add PdgCode {-16}

}

#####################
# MC truth jet finder
#####################

module FastJetFinder GenJetFinderAK8 {
  set InputArray NeutrinoFilter/filteredParticles

  set OutputArray jetsAK8

  # algorithm: 1 CDFJetClu, 2 MidPoint, 3 SIScone, 4 kt, 5 Cambridge/Aachen, 6 antikt
  set JetAlgorithm 6
  set ParameterR 0.8

  set JetPTMin 200.0
}


#############
# Rho pile-up
#############

module FastJetFinder Rho {
#  set InputArray Calorimeter/towers
  set InputArray EFlowMergerCHS/eflow

  set ComputeRho true
  set RhoOutputArray rho

  # area algorithm: 0 Do not compute area, 1 Active area explicit ghosts, 2 One ghost passive area, 3 Passive area, 4 Voronoi, 5 Active area
  set AreaAlgorithm 5

  # jet algorithm: 1 CDFJetClu, 2 MidPoint, 3 SIScone, 4 kt, 5 Cambridge/Aachen, 6 antikt
  set JetAlgorithm 4
  set ParameterR .8
#0.4
  set GhostEtaMax 5.0

  add RhoEtaRange -5.0 -4.0
  add RhoEtaRange -4.0 -1.5
  add RhoEtaRange -1.5 1.5
  add RhoEtaRange 1.5 4.0
  add RhoEtaRange 4.0 5.0

  set JetPTMin 0.0
}


##############
# Jet finder
##############

module FastJetFinder FastJetFinderAK8 {
#  set InputArray TowerMerger/towers
  set InputArray EFlowMergerCHS/eflow

  set OutputArray jets

  set AreaAlgorithm 5

  # algorithm: 1 CDFJetClu, 2 MidPoint, 3 SIScone, 4 kt, 5 Cambridge/Aachen, 6 antikt
  set JetAlgorithm 6
  set ParameterR 0.8

  set ComputeNsubjettiness 1
  set Beta 1.0
  set AxisMode 4

  set ComputeTrimming 1
  set RTrim 0.2
  set PtFracTrim 0.05

  set ComputePruning 1
  set ZcutPrun 0.1
  set RcutPrun 0.5
  set RPrun 0.8

  set ComputeSoftDrop 1
  set BetaSoftDrop 0.0
  set SymmetryCutSoftDrop 0.1
  set R0SoftDrop 0.8

  set JetPTMin 200.0
}

###########################
# Jet Pile-Up Subtraction
###########################

module JetPileUpSubtractor JetPileUpSubtractor {
  set JetInputArray FastJetFinder/jets
  set RhoInputArray Rho/rho

  set OutputArray jets

  set JetPTMin 15.0
}

##############################
# Jet Pile-Up Subtraction AK8
##############################

module JetPileUpSubtractor JetPileUpSubtractorAK8 {
  set JetInputArray FastJetFinderAK8/jets
  set RhoInputArray Rho/rho

  set OutputArray jets

  set JetPTMin 15.0
}


##################
# Jet Energy Scale
##################

module EnergyScale JetEnergyScaleAK8 {
  set InputArray JetPileUpSubtractorAK8/jets
  set OutputArray jets

 # scale formula for jets
  set ScaleFormula {1.00}
}

########################
# Jet Flavor Association
########################

module JetFlavorAssociation JetFlavorAssociationAK8 {

  set PartonInputArray Delphes/partons
  set ParticleInputArray Delphes/allParticles
  set ParticleLHEFInputArray Delphes/allParticlesLHEF
  set JetInputArray JetEnergyScaleAK8/jets

  set DeltaR 0.8
  set PartonPTMin 100.0
  set PartonEtaMax 4.0

}

###############################################################################################################
# StatusPidFilter: this module removes all generated particles except electrons, muons, taus, and status == 3 #
###############################################################################################################

module StatusPidFilter GenParticleFilter {

    set InputArray Delphes/allParticles
    set OutputArray filteredParticles
    set PTMin 5.0

}





##################
# ROOT tree writer
##################

module TreeWriter TreeWriter {
# add Branch InputArray BranchName BranchClass
  add Branch GenParticleFilter/filteredParticles Particle GenParticle
  add Branch PileUpMerger/vertices Vertex Vertex

  add Branch GenJetFinderAK8/jetsAK8 GenJetAK8 Jet
# add Branch Calorimeter/eflowTracks EFlowTrack Track
  # NOT REALLY SURE IF THIS PART WILL WORK
  add Branch RecoPuFilter/eflowTracks EFlowTrack Track
  add Branch NeutralEFlowMerger/eflowTowers EFlowTower Tower
  add Branch EFlowMergerCHS/eflow EFlowTower Tower
  add Branch Calorimeter/towers CaloTower Tower
#  add Branch Calorimeter/edges CaloEdges Tower

  add Branch JetEnergyScaleAK8/jets JetAK8 Jet

}
