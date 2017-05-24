import numpy as np
from viz import *
from likelihood import *


PLOT_DIR = '/user_data/nelson/training_output/train0PU_test0PU/'
network_eval= 'train0PU_test0PU.npy'
data = np.load('../../../Jetinfo0PU_noimage.npz')

signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau21']

signal_pct = data['signal'].mean()
print ('{} % signal'.format(signal_pct))

signal = (signal == 1)
background = (signal == False)

plt.figure()
# -- plot some kinematics...
n1, _, _ = plt.hist(pt[signal], bins=np.linspace(200, 300, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", linewidth=2)
n2, _, _ = plt.hist(pt[background], bins=np.linspace(200, 300, 100), histtype='step', color='blue', label='QCD', linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.xlabel(r'$p_T$ [GeV]')
plt.ylabel('Count')
plt.ylim(0, 1.2 * mx)
plt.title(r'Jet $p_T$ distribution, $p_T \in [200, 300]$ GeV' + '\n' + 
	r'$m_{\mathsf{jet}}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR+'unweighted-pt-distribution-[250-300].pdf')
plt.show()


# -- calculate the weights
weights = np.ones(data['image'].shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

weights[signal] = get_weights(reference_distribution, pt[signal], 
	bins=np.linspace(200, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
	bins=np.linspace(200, 300, 200))

plt.figure()
# -- plot reweighted...
plt.hist(pt[signal], bins=np.linspace(200, 300, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal], linewidth=2)
plt.hist(pt[background], bins=np.linspace(200, 300, 100), histtype='step', color='blue', label='QCD', weights=weights[background], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlabel(r'$p_T$ (GeV)')
plt.ylabel('Count')
plt.title(r'Weighted Jet $p_T$ distribution (matched to QCD)')
plt.legend()
         
plt.savefig(PLOT_DIR+'weighted-pt-distribution[200-300].pdf')
plt.show()


plt.figure()
# -- plot weighted mass
n1, _, _ = plt.hist(mass[signal], bins=np.linspace(65, 95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal], linewidth=2)
n2, _, _ = plt.hist(mass[background], bins=np.linspace(65, 95, 100), histtype='step', color='blue', label='QCD', weights=weights[background], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlabel(r'Jet $m$ [GeV]')
plt.ylabel('Weighted Count')
plt.title(r'Weighted Jet $m$ Distribution ($p_T \in [200, 300]$ GeV, matched to QCD)' + '\n' + 
	r'$m_{\mathsf{jet}}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR+'weighted-mass-distribution[200-300].pdf')
plt.show()


plt.figure()
# -- plot weighted tau_21
n1, _, _ = plt.hist(tau_21[signal], bins=np.linspace(0, 0.95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal], linewidth=2)
n2, _, _ = plt.hist(tau_21[background], bins=np.linspace(0, 0.95, 100), histtype='step', color='blue', label='QCD', weights=weights[background], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlabel(r'Jet $\tau_{21}$')
plt.ylabel('Weighted Count')
plt.title(r'Weighted Jet $\tau_{21}$ distribution ($p_T \in [200, 300]$ GeV, matched to QCD)' + '\n' + 
	r'$m_{\mathsf{jet}}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR+'weighted-tau21-distribution[200-300].pdf')
plt.show()
print("data plots done")

# -- likelihood
# mass_bins = np.linspace(64.99, 95.01, 10)#, np.linspace(35, 159.99, 30), np.array([160, 900])))
# tau_bins = np.concatenate((np.array([0, 0.1]), np.linspace(0.1001, 0.7999999999, 10), np.array([0.8, 1])))
# P_2d = Likelihood2D(mass_bins, tau_bins)
# P_2d.fit((mass[signal], tau_21[signal]), (mass[background], tau_21[background]), weights=(weights[signal], weights[background]))
# P_2d.fit((mass[signal], tau_21[signal]), (mass[background], tau_21[background]), weights=(weights[signal], weights[background]))
# mass_nsj_likelihood = P_2d.predict((mass, tau_21))
# log_likelihood = np.log(mass_nsj_likelihood)
# -- plot weighted mass + nsj likelihood
# plt.hist((log_likelihood[signal == True]), bins=np.linspace(-3.6, 6.5, 20), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal])
# plt.hist((log_likelihood[signal == False]), bins=np.linspace(-3.6, 6.5, 20), histtype='step', color='blue', label='QCD')
# plt.xlabel(r'$\log(P(\mathrm{signal}) / P(\mathrm{background}))$')
# plt.ylabel('Weighted Count')
# plt.title(r'Weighted Jet $m, \tau_{21}$ likelihood distribution ($p_T$ matched to QCD)')
# plt.legend()
# #plt.savefig(PLOT_DIR+PLOT_DIR % 'weighted-mass-nsj-likelihood-distribution.pdf')
#plt.show()


plt.figure(figsize=(20,20),dpi=200)
plt.rc('font', size=30)
plt.rc('legend', fontsize=30)

# -- plot DL out
y_dl = np.load(network_eval)
n1, _, _ = plt.hist(y_dl[signal], bins=np.linspace(0, 1, 100), histtype='step', log=True,color='red',label=r"$W' \rightarrow WZ$", weights=weights[signal], linewidth=2)
n2, _, _ = plt.hist(y_dl[background], bins=np.linspace(0, 1, 100), histtype='step', log=True,color='blue',label='QCD', weights=weights[background], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlim(0,1)
plt.xlabel(r'Deep Network Output')
plt.ylabel('')
plt.title(r'Weighted Deep Network Distribution ($p_T \in [200, 300]$ matched to QCD)' + '\n' + 
          r'$m_{\mathsf{jet}}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR + 'weighted-deep-net-distribution.pdf')
plt.show()

# DL_lh = Likelihood1D(np.linspace(0, 1, 60))
# DL_lh.fit(y_dl[signal], y_dl[background], weights=(weights[signal], weights[background]))
# DLlikelihood = DL_lh.predict(y_dl)

# -- plot DL output


# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(n_jobs=2)

# df = np.zeros((y_dl.shape[0], 3))
# df[:, 0] = y_dl
# df[:, 1] = mass
# df[:, 2] = tau_21
# rf.fit(df[:n_train], y_[:n_train], sample_weight=weights[:n_train])


# y_rf = rf.predict_proba(df[n_train:])


# from sklearn.ensemble import GradientBoostingClassifier


# lh_bins = np.linspace(-4, 6, 4)#, np.linspace(35, 159.99, 30), np.array([160, 900])))
# # lh_bins = np.concatenate((np.array([0, 0.1]), np.linspace(0.1001, 0.7999999999, 10), np.array([0.8, 1])))
# dnn_bins = np.linspace(0, 1, 100)
# CLH = Likelihood2D(lh_bins, dnn_bins)
# CLH.fit((tau_21[signal], y_dl[signal]), (log_likelihood[background], y_dl[background]), weights=(weights[signal], weights[background]))
# CLH.fit((log_likelihood[signal], y_dl[signal]), (log_likelihood[background], y_dl[background]), weights=(weights[signal], weights[background]))
# combined_likelihood = CLH.predict((log_likelihood, y_dl))
# # log_likelihood = np.log(mass_nsj_likelihood)


import matplotlib.cm as cm

X_ = data['image'].reshape((data['image'].shape[0], 25**2))

plt.figure(figsize=(15,12))
plt.rc('font', size=25)
plt.rc('legend', fontsize=25)
zr=np.zeros(25**2)
for i in range(625): 
    print(i) 
    zr[i] = np.corrcoef(X_[:, i], y_dl)[0, 1]
rec = zr.reshape((25, 25))
rec[np.isnan(rec)] = 0.0
# plt.imshow(rec, interpolation='nearest', cmap=custom_div_cmap(101), vmax = np.max(np.abs(rec)), vmin=-np.max(np.abs(rec)), extent=[-1,1,-1,1])
plt.imshow(rec, interpolation='nearest', cmap=cm.seismic, vmax = .32, vmin=-.32, extent=[-1,1,-1,1])
# plt.axis('off')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Pearson Correlation Coefficient')
plt.title(r'Correlation of Deep Network output with pixel activations.' + '\n' + 
	r'$p_T^W \in [200, 300]$ matched to QCD, $m_{W}\in [65, 95]$ GeV')
plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')

plt.savefig(PLOT_DIR+'pixel-activations-corr.pdf')
plt.show()

tau_windows = [(0.19, 0.21), (0.43, 0.44), (0.7, 0.72)]
mass_windows = [(65, 67), (79, 81), (93, 95)]

windows = []

import itertools

for tw, mw in itertools.product(tau_windows, mass_windows):
	print 'working on nsj in [{}, {}] and mass in [{}, {}]'.format(*list(itertools.chain(tw, mw)))
	print 'selection of cube'
	small_window = (tau_21 < tw[1]) & (tau_21 > tw[0]) & (mass > mw[0]) & (mass < mw[1])
	print 'selection of s'
	swsignal = small_window & signal
	print 'selection of b'
	swbackground = small_window & background
	print 'plotting difference'
	rec = np.average(data['image'][swsignal], weights=weights[swsignal], axis=0) - np.average(data['image'][swbackground], axis=0)
	windows.append({'image' : rec, 'nsj' : tw, 'mass' : mw})
	# plt.imshow(rec, interpolation='nearest', cmap=custom_div_cmap(101), vmax = np.max(np.abs(rec)), vmin=-np.max(np.abs(rec)), extent=[-1,1,-1,1])
	# cb = plt.colorbar()
	# cb.ax.set_ylabel(r'$\Delta E_{\mathsf{normed}}$ deposition')
	# plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
	# plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
	# idfr = [mw[0], mw[1], tw[0], tw[1]]
	# plt.title(
	# 	r"Difference in per pixel normalized energy deposition" + 
	# 	'\n' + 
	# 	r"between $W' \rightarrow WZ$ and QCD in $m \in [%.2f, %.2f]$ GeV, $\tau_{21}\in [%.2f, %.2f]$ window." % (idfr[0], idfr[1], idfr[2], idfr[3]))
	# #plt.savefig(PLOT_DIR % 'im-diff-m{}-{}-nsj.{}.{}.pdf'.format(idfr[0], idfr[1], idfr[2], idfr[3]))
	# plt.show()

max_diff = np.max([np.percentile(np.abs(w['image']), 99.99) for w in windows])

#model
for w in windows:
	rec = w['image']
	plt.imshow(rec, interpolation='nearest', cmap=custom_div_cmap(101), vmax = max_diff, vmin=-max_diff, extent=[-1,1,-1,1])
	cb = plt.colorbar()
	cb.ax.set_ylabel(r'$\Delta E_{\mathsf{normed}}$ deposition')
	plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
	plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
	plt.title(
		r"Difference in per pixel normalized energy deposition between" + 
		'\n' + 
		r"$W' \rightarrow WZ$ and QCD in $m \in [%i, %i]$ GeV, $\tau_{21}\in [%.2f, %.2f]$ window." % (int(w['mass'][0]), int(w['mass'][1]), w['nsj'][0], w['nsj'][1]))
	plt.savefig(PLOT_DIR+'new-im-diff-m{}-{}-nsj.{}.{}.pdf'.format(int(w['mass'][0]), int(w['mass'][1]), w['nsj'][0], w['nsj'][1]))
	plt.show()

