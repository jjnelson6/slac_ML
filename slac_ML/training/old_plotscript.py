###Extra plots if you are interested( may have to work out a few bugs)
"""
window = (tau_21 < 0.8) & (tau_21 > 0.2)
pt, mass, tau_21, signal, bakground, y_dl = pt[window], mass[window], tau_21[window], signal[window], background[window], y_dl[window],

n_obs = int(window.sum())

ref = np.zeros((n_obs, 3))
ref[:, 0] = pt
ref[:, 1] = mass
ref[:, 2] = tau_21

cube = np.zeros((n_obs / 2, 3))

cube[:, 0] = np.random.uniform(250, 300, n_obs / 2)
cube[:, 1] = np.random.uniform(65, 95, n_obs / 2)
cube[:, 2] = np.random.uniform(0.2, 0.8, n_obs / 2)


binning = (
		np.linspace(250, 300, 20),
		np.linspace(65, 95, 19),
		# np.concatenate(
				# (
					# np.array([0, 0.2]), 
					np.linspace(0.2, 0.8, 19)
					# np.array([0, 0.1, 0.2, 0.4, 0.55, 0.7, 1])
				# )
			# )
		)



# H_s, _ = np.histogramdd(ref[signal], bins=binning, normed=False)
# H_b, _ = np.histogramdd(ref[background], bins=binning, normed=False)
# H_ref, _ = np.histogramdd(cube, bins=binning, normed=False)

# flat_cube = H_ref / H_s

class NDWeights(object):
	#docstring for NDWeights
	def __init__(self, bins):
		super(NDWeights, self).__init__()
		self.bins = bins
	def fit(self, X, truth, reference):
		H_s, _ = np.histogramdd(X[truth == 1], bins=self.bins, normed=False)
		H_b, _ = np.histogramdd(X[truth == 0], bins=self.bins, normed=False)
		H_ref, _ = np.histogramdd(reference, bins=self.bins, normed=False)
		self.flat_cube_s = H_ref / H_s
		self.flat_cube_b = H_ref / H_b

	def predict(self, X, truth):
		ix = [(self.bins[i].searchsorted(X[:, i]) - 1) for i in xrange(len(self.bins))]
		ix = np.array(ix).T
		print ix
		weights = []
		for i, label in zip(ix, truth):
			if label == 1:
				w = np.copy(self.flat_cube_s[i[0]])
			else:
				w = np.copy(self.flat_cube_b[i[0]])
			for j in xrange(1, len(self.bins)):
				w = w[i[j]]
			weights.append(w)
		weights = np.array(weights)
		weights[np.isinf(weights)] = weights[np.isfinite(weights)].max()
		return weights


ndweights = NDWeights(binning)
ndweights.fit(ref, signal, cube)


cube_weights = ndweights.predict(ref, signal)
cube_weights[np.isinf(cube_weights)] = cube_weights[np.isfinite(cube_weights)].max()





# -- plot reweighted...
n1, _, _ = plt.hist(pt[signal], bins=np.linspace(250, 300, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal], linewidth=2)
n2, _, _ = plt.hist(pt[signal == False], bins=np.linspace(250, 300, 100), histtype='step', color='blue', label='QCD', weights=cube_weights[signal==False], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.xlabel(r'$p_T$ (GeV)')
plt.ylabel('Count')
plt.ylim(0, 1.2 * mx)
plt.title(r'Weighted Jet $p_T$ distribution in $(p_T, m, \tau_{21})$ flat hypercube')
plt.legend()
         
plt.savefig(PLOT_DIR+'weighted-pt-distribution[200-300]-cube.pdf')
plt.show()



# -- plot weighted mass
n1, _, _ = plt.hist(mass[signal == True], bins=np.linspace(65, 95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal], linewidth=2)
n2, _, _ = plt.hist(mass[signal == False], bins=np.linspace(65, 95, 100), histtype='step', color='blue', label='QCD', weights=cube_weights[signal==False], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlabel(r'Jet $m$ [GeV]')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m$ distribution in $(p_T, m, \tau_{21})$ flat hypercube')
plt.legend()
plt.savefig(PLOT_DIR+'weighted-mass-distribution[200-300]-cube.pdf')
plt.show()



# -- plot weighted tau_21
n1, _, _ = plt.hist(tau_21[signal == True], bins=np.linspace(0.2, 0.8, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal], linewidth=2)
n2, _, _ = plt.hist(tau_21[signal == False], bins=np.linspace(0.2, 0.8, 100), histtype='step', color='blue', label='QCD', weights=cube_weights[signal==False], linewidth=2)
mx = max(np.max(n1), np.max(n2))
plt.ylim(0, 1.2 * mx)
plt.xlabel(r'Jet $\tau_{21}$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $\tau_{21}$ distribution in $(p_T, m, \tau_{21})$ flat hypercube')
plt.legend()
plt.savefig(PLOT_DIR+'weighted-tau21-distribution[200-300]-cube.pdf')
plt.show()

# P_mass = Likelihood1D(np.concatenate((np.array([0, 25]), np.linspace(25, 35, 5), np.linspace(35, 160, 20), np.array([160, 900]))))
# P_mass.fit(mass[:n_train][signal[:n_train] == 1], mass[:n_train][signal[:n_train] == 0], weights=(cube_weights[:n_train][signal[:n_train] == 1], cube_weights[:n_train][signal[:n_train] == 0]))
# mass_likelihood = P_mass.predict(mass)

# # -- plot weighted mass likelihood
# plt.hist((mass_likelihood[signal == True]), bins=np.linspace(0, 10, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal])
# plt.hist((mass_likelihood[signal == False]), bins=np.linspace(0, 10, 100), histtype='step', color='blue', label='QCD')
# plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
# plt.ylabel('Count')
# plt.title(r'Weighted Jet $m$ likelihood distribution ($p_T$ matched to QCD)')
# plt.legend()
# #plt.savefig(PLOT_DIR % 'weighted-mass-likelihood-distribution.pdf')
plt.show()




mass_bins = np.linspace(64.99, 95.01, 10)#, np.linspace(35, 159.99, 30), np.array([160, 900])))
tau_bins = np.linspace(0.2, 0.8, 10)
P_2d = Likelihood2D(mass_bins, tau_bins)
P_2d.fit((mass[signal], tau_21[signal]), (mass[signal == False], tau_21[signal == False]), weights=(cube_weights[signal], cube_weights[signal == False]))
P_2d.fit((mass[signal], tau_21[signal]), (mass[signal == False], tau_21[signal == False]), weights=(cube_weights[signal], cube_weights[signal == False]))
mass_nsj_likelihood = P_2d.predict((mass, tau_21))
log_likelihood = np.log(mass_nsj_likelihood)
# -- plot weighted mass + nsj likelihood
# plt.hist((log_likelihood[signal == True]), bins=np.linspace(-0.5, 0.5, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal], linewidth=2)
# plt.hist((log_likelihood[signal == False]), bins=np.linspace(-0.5, 0.5, 100), histtype='step', color='blue', label='QCD', weights=cube_weights[background], linewidth=2)
# plt.xlabel(r'$\log P(\mathrm{signal}) / P(\mathrm{background})$')
# plt.ylabel('Count')
# plt.title(r'Weighted Jet $m, \tau_{21}$ likelihood distribution in $(p_T, m, \tau_{21})$ flat hypercube')
# plt.legend()
#plt.savefig(PLOT_DIR+'weighted-mass-nsj-likelihood-distribution-cube.pdf')
#plt.show()



# y_dl = np.load('./yhat.npy')

# # -- plot DL output
# plt.hist(y_dl[signal == True], bins=np.linspace(0, 1, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=cube_weights[signal], linewidth=2)
# plt.hist(y_dl[signal == False], bins=np.linspace(0, 1, 100), histtype='step', color='blue', label='QCD', weights=cube_weights[background], linewidth=2)
# # plt.ylim(0, 82000)
# plt.xlabel(r'Deep Network Output')
# plt.ylabel('Count')
# plt.title(r'Weighted Deep Network distribution in $(p_T, m, \tau_{21})$ flat hypercube')
# plt.legend()
# #plt.savefig(PLOT_DIR % 'weighted-deep-net-distribution-cube.pdf')
plt.show()
"""

"""
cube_discs = {}
add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal, 2-tau_21, weights=cube_weights), cube_discs)
add_curve(r'Deep Network, trained on $p_T \in [250, 300]$ GeV outside cube', 'red', calculate_roc(signal, y_dl, weights=cube_weights, bins = 1000000), cube_discs)
add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood)', 'blue', calculate_roc(signal, (log_likelihood), bins=1000000, weights=cube_weights), cube_discs)
fg = ROC_plotter(cube_discs, title=r"$W' \rightarrow WZ$ vs. QCD $(p_T, m, \tau_{21})$ flat hypercube" + '\n' + 
	r'$m\in [65, 95]$ GeV, $p_T\in [250, 300]$ GeV, $\tau_{21}\in [0.2, 0.8]$.', min_eff = 0.2, max_eff=0.8, logscale=False)
fg.savefig(PLOT_DIR+'roc-cube-outside.pdf')

plt.show()





sel = (data['tau_21'] > 0.2) & (data['tau_21'] < 0.8)

X_cube = data['image'][sel].reshape((sel.sum(), 25**2)).astype('float32')
# X_ = X_ * data['total_energy'][:, np.newaxis]
y_cube = data['signal'][sel].astype('float32')

#print X_cube.shape



# -- build the model
dl = Sequential()
# dl.add(Merge([raw, gaussian], mode='concat'))
# dl.add(Dense(1000, 512))
# dl.add(Dropout(0.1))
dl.add(MaxoutDense(256, 5, input_shape=(625,),init='he_normal'))

dl.add(Dropout(0.3))
dl.add(MaxoutDense(128, 5, init='he_normal'))


dl.add(Dropout(0.1))
dl.add(Dense(64))
dl.add(Activation('relu'))
dl.add(Dropout(0.3))
dl.add(Dense(25))
dl.add(Activation('relu'))
dl.add(Dropout(0.1))
dl.add(Dense(1))
dl.add(Activation('sigmoid'))

dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dl.load_weights(we_str)
# try:
# 	print 'Training!'
# 	h = dl.fit(X_cube, y_cube, batch_size=28, nb_epoch=20, show_accuracy=True, 
# 	               validation_split=0.5, 
# 	               callbacks = [
# 	                   EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
# 	                   ModelCheckpoint('./SLACNet-cube.h5', monitor='val_loss', verbose=True, save_best_only=True)
# 	               ], 
# 	               sample_weight=np.sqrt(cube_weights))
# except KeyboardInterrupt:
# 	print 'stop'

y_dl_cube = dl.predict(X_cube, verbose=True, batch_size=200).ravel()



def normalize_rows(x):
    def norm1d(a):
        return a / a.sum()
    x = np.array([norm1d(r) for r in x])
    return x

H, b_x, b_y = np.histogram2d(
	mass[(signal == False)], 
	y_dl_cube[(signal == False)], 
	bins=(np.linspace(65, 95, 35), np.linspace(0, 1, 35)), 
	normed=True)


plt.imshow(np.flipud(normalize_rows(H.T)), extent=(65, 95, 0, 1), aspect='auto', interpolation='nearest')
plt.xlabel('QCD Jet Mass [GeV]')
plt.ylabel(r'Deep Network output')
plt.title(r'PDF of QCD Jet Mass, binned vs. Deep Network output' + '\n' + 
	r'Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [65, 95]$ GeV')
cb = plt.colorbar()
cb.set_label(r'$P(\mathrm{mass} \vert \hat{y})$')
plt.savefig(PLOT_DIR+'mass-dist-yhat-unweighted.pdf')
plt.show()





cube_discs = {}
add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal, 2-tau_21, weights=cube_weights), cube_discs)
add_curve(r'Deep Network, trained on $p_T \in [250, 300]$ GeV outside cube', 'red', calculate_roc(signal, y_dl, weights=cube_weights, bins = 1000000), cube_discs)
add_curve(r'Deep Network, trained on $p_T \in [250, 300]$ GeV inside cube', 'purple', calculate_roc(signal, y_dl_cube, weights=cube_weights, bins = 1000000), cube_discs)
add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood)', 'blue', calculate_roc(signal, (log_likelihood), bins=1000000, weights=cube_weights), cube_discs)
fg = ROC_plotter(cube_discs, title=r"$W' \rightarrow WZ$ vs. QCD $(p_T, m, \tau_{21})$ flat hypercube" + '\n' + 
	r'$m\in [65, 95]$ GeV, $p_T\in [250, 300]$ GeV, $\tau_{21}\in [0.2, 0.8]$.', min_eff = 0.2, max_eff=0.8, logscale=False)
fg.savefig(PLOT_DIR+'roc-cube-inside.pdf')

plt.show()





# print 'loading window data'
# data_ben = np.load('../ben-additions/ben-window.npy')

# print 'generating model from yaml'
# # -- build the model
# dl = model_from_yaml('./SLACNetBFboringNet2-final.yaml')

# print 'compiling...'
# dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
# print 'loading weights...'
# dl.load_weights('./SLACNetBFboringNet2-final-roc.h5')

# X_ben = data_ben['image'].reshape((data_ben.shape[0], 25**2))

# y_dl_ben = dl.predict(X_ben, verbose=True, batch_size=200).ravel()

# ben_likelihood = P_2d.predict((data_ben['jet_mass'], data_ben['tau_21']))

# signal_ben = data_ben['signal'] == 1


# print 'estimating LDA'
# from sklearn.lda import LDA
# clf = LDA()
# ylda = clf.fit_transform(X_ben, data_ben['signal'])

# discs = {}
# add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal_ben, 2-data_ben['tau_21']), discs)
# add_curve(r'Deep Network, trained on $p_T \in [250, 300]$ GeV', 'red', calculate_roc(signal_ben, y_dl_ben, bins=1000000), discs)
# add_curve(r'LDA inside window', 'green', calculate_roc(signal_ben, (ylda)), discs)
# add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood, outside window)', 'blue', calculate_roc(signal_ben, (ben_likelihood), bins=1000000), discs)




# fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD $p_T \in [250, 255]$ GeV" + '\n' + 
# 	r'$m_{\mathsf{jet}}\in [79, 81]$ GeV, $\tau_{21} \in [0.19, 0.21]$', min_eff = 0.2, max_eff=0.8, logscale=False)
# fg.savefig(PLOT_DIR % 'small-window-combined-roc.pdf')

# plt.show()





# from statsmodels.stats.weightstats import DescrStatsW
# ds = DescrStatsW(x, weights=weights)
"""
