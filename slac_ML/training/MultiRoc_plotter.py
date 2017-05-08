import numpy as np
from matplotlib.colors import LogNorm
from viz import *
from likelihood import *


PLOT_DIR = ''#'../../preprocessing'
network_eval= ['../../train0PU_test0PU.npy','../../train35PU_test35PU.npy','../../train140PU_test140PU.npy','../../train200PU_test200PU.npy']
#['../../train0PU_test0PU.npy','../../train0PU_test35PU.npy','../../train0PU_test140PU.npy']
filen= ['../../preprocessing/Jetinfo0PU_noimage.npz','../../preprocessing/Jetinfo35PU_noimage.npz','../../preprocessing/Jetinfo140PU_noimage.npz'  ,'../../preprocessing/Jetinfo200PU_noimage.npz']
#['../../preprocessing/Jetinfo0PU_noimage.npz','../../preprocessing/Jetimages_35PU.npz','../../preprocessing/Jetimages_140PU.npz']
name=['0PU','35PU','140PU','200PU']
color=['blue','black','red','green']

count=0
discs = {}
 
for f in filen:
    data = np.load(f)
    
    signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau21']
    signal_pct = data['signal'].mean()
    print '{}% signal'.format(signal_pct)

    signal = (signal == 1)
    background = (signal == False)

# -- calculate the weights
    weights = np.ones(data['signal'].shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
    reference_distribution = pt[background]

    weights[signal] = get_weights(reference_distribution, pt[signal], 
                              bins=np.linspace(200, 300, 200))

    weights[background] = get_weights(reference_distribution, pt[background], 
                                      bins=np.linspace(200, 300, 200))

    y_dl = np.load(network_eval[count])
       
    add_curve(r'$\tau_{21}$' +name[count],color[count], calculate_roc(signal, 2-tau_21,weights=weights), '--',discs)
    add_curve(r'Deep Network,'+ name[count] +'-trained and '+ name[count]+'-tested on $p_T \in [200, 300]$ GeV',color[count],calculate_roc(signal, y_dl, weights=weights, bins=1000000),'-' ,discs)
   


    a=calculate_roc(signal, y_dl, weights=weights, bins=1000000)
    
    print 'signal effciency:', a[0][a[0]<.25][-1]
    print 'background effciency:', 1/a[1][a[0]<.25][-1]
    
    print 'signal effciency:', a[0][a[0]<.50][-1]
    print 'background effciency:', 1/a[1][a[0]<.50][-1]

    print 'signal effciency:', a[0][a[0]<.75][-1]
    print 'background effciency:', 1/a[1][a[0]<.75][-1]


    count+=1
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD ($p_T \in [200, 300]$ GeV, matched to QCD)" + '\n' + 
                     r'$m_{\mathsf{jet}}\in [65, 95]$ GeV', min_eff = 0.2, max_eff=0.8,logscale=False)
   
fg.savefig(PLOT_DIR+'everything.png')
plt.show()
