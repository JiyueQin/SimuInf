def confset(est, lower, upper, threshold=0):
    """
    Calculate inner and outer confidence sets by inverting the simutaneous confidence bands.

    Parameters
    ----------
    est: array
      estimate
    lower: array
      lower limits of the SCB
    upper: array
      upper limits of the SCB
    threshold : int
      threshold to be used for sub-setting

    Returns
    -------
    est_set : array(Boolean)
      voxels in the Ac_hat area
    inner_set : array(Boolean)
      voxels in the lower confidence set
    outer_set : array(Boolean)
      voxels in the upper confidence set
    plot_add : array(int)
      area representing lower_set + upper_set + Achat
    n_rej : int or list
      number of voxels rejected by the procedure

    Example
    -------
    data = np.random.normal(size=(30,40,50))
    est, lower, upper = confband(data, m_boots=100, method='multi_regular', multiplier = 'g')
    est_set, inner_set, outer_set = confset(est, lower, upper)
    plt.imshow(inner_set)
    plt.show()
    plt.imshow(est_set)
    plt.show()
    plt.imshow(outer_set)
    plt.show()
    """

    est_set = est >= threshold
    inner_set = lower >= threshold
    outer_set = upper >= threshold
    return est_set, inner_set, outer_set




from crtoolbox.generate import generate_CRs
from crtoolbox.lib.regression import regression
from nilearn.image import get_data
import os 
import numpy as np 
def sss(out_dir, data_dir=None, n=None, m_boots=1000, threshold = 2, alpha = 0.05, mask = None, three_d = False, load_only = False):
    # load_only: True if confidence set results are already computed in the output folder and you want to directly load them
    if not load_only:
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
              if os.path.isfile(os.path.join(data_dir, f))]
        # n: sample size 
        X = np.ones((n,1))
        # Fit the regression model
        muhat_file, sigma_file, resid_files = regression(data_files, X, out_dir)
        # returns a tuple of (inner_file, outer_file, est_file, quantile_est)
        CR_files = generate_CRs(muhat_file, sigma_file, resid_files, out_dir, threshold, 1-alpha, n_boot=m_boots, mask = mask)
        # match the order of confset (est, inner, outer)
        CR_files = [CR_files[2], CR_files[0], CR_files[1]]
    else:
        CR_files =  [os.path.join(out_dir, f) for f in ['Estimated_Ac.nii', 'Upper_CR_0.95.nii', 'Lower_CR_0.95.nii']]
    if three_d:
        confset = tuple(get_data(fname)[...,0] for fname in CR_files)
    else:
        confset = tuple(np.load(fname) for fname in CR_files)
    return confset