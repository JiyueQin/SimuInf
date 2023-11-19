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
