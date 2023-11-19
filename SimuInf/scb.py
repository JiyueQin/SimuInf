import numpy as np
import sys


def bootstrap(data, alpha=0.05, m_boots=5000, boot_data_type='res', boot_type='multiplier',
              standardize='t', multiplier='r'):
    """
    Compute bootstrap estimator for the quantile of the maximum of a random field.

    Parameters
    ----------
    data: array(int or float)
      array of dimension K_1 x ... x K_D x N
      containing N-realizations or N-residuals of a random field over a D-dimensional domain
    alpha: float, optional, default: 0.05
      1-alpha is the targeted covering rate of the simultaneous confidence interval
    m_boots: int, optional, default: 5e3
      number of bootstrap samples
    boot_data_type: string, optional, default: "res"
      the type of the input data.
      Options are "res" for residuals, "obs" for observations, see details.
    boot_type: string, optional, default: "multiplier"
      the type of bootstrap
      Options are "multiplier" for multiplier bootstrap and "nonparametric" for nonparametric bootstrap.
    standardize: string, optional, default: "t"
      the type of standardization
      Options are "t" for t score(use bootstrap sd) and "z" for z score(only use sample sd).
    multiplier: string, optional, default: "r"
      the type of multiplier for multiplier bootstrap, only used when boot_type="multiplier"
      Options are "r" for Rademacher multipliers and "g" for Gaussian multipliers.

    Returns
    -------
    tuple with elements:
    dist: list
      realizations of maximum of the random field in bootstrap
    q: float
      alpha-quantile value of dist

    Details
    -------
    A particular bootstrap method can be specified based the following three aspects:
      1. what to bootstrap on:
         residuals(boot_data_type='res') or original observations(boot_data_type='obs').
      2. how to generate bootstrap samples:
         resample with replacement(ie, nonparametric bootstrap, boot_type='nonparametric') or
         use multipliers (ie, multiplier bootstrap, boot_type='multiplier')
      3. how to standardize:
         t score(use bootstrap sd, standardize= t') or z score(only use sample sd, standardize='z')

    For example:
    - the non-parametric bootstrap in Degras (2011)
      "Simultaneous confidence bands for non-parametric regression with functional data" can be specified with:
        boot_data_type='obs', boot_type='nonparametric', standardize='z'
    - the multiplier bootstrap in Cherno can be specified with:
        boot_data_type='res', boot_type='multiplier', standardize = 'z', multiplier='r'(or 'g'),
    - the multiplier-t bootstrap in Telschow Schwartzmann (2019)
      "Simultaneous confidence bands for non-parametric regression with functional data" can be specified with:
        boot_data_type='res', boot_type='multiplier', standardize = 't', multiplier = 'r'(or 'g')

    Usually, holding other aspects the same, the z- and the t-standardization agree for large sample sizes.
    However, for small sample sizes the t-standardization has better covering rates,
    but higher variability in the estimate of the quantile.

    Example
    -------
    data = np.random.normal(size=(30,40,50))
    bootstrap(data)
    bootstrap(data, m_boots=100, boot_data_type = 'obs', boot_type='multiplier', multiplier='g')

    """

    # -----check user inputs
    if not isinstance(m_boots, int):
        sys.exit("The input m_boots needs to be a positive natural number.")

    # -----computation
    data_dim = data.shape
    nsubj = data_dim[-1]
    d = len(data_dim) - 1
    boot_stat_dim = (*data_dim[:-1], m_boots)
    concat_data = data.reshape((np.prod(data_dim[:-1]), nsubj))

    # get sample mean  should this be 0 for all kinds of residuals?
    sample_mean = data.mean(axis=-1)

    if boot_type in ['nonparametric']:
        # counter: array of shape (n_subj, m_boots), for bootstrap resampling
        counter = np.random.multinomial(nsubj, np.repeat(1 / nsubj, nsubj),
                                        m_boots).transpose()
        # get bootstrap mean
        boot_means = (concat_data @ counter).reshape(boot_stat_dim) / nsubj
        # get sample or bootstrap standard deviation sigma
        if standardize == 'z':
            sigma = data.std(axis=-1, ddof=1).repeat(m_boots).reshape(boot_stat_dim)
        elif standardize == 't':
            boot_mean_squared = (concat_data ** 2 @ counter).reshape(boot_stat_dim) / nsubj
            sigma = np.sqrt(np.abs(boot_mean_squared - boot_means ** 2) * (nsubj / (nsubj - 1)))
        # compute bootstrap distribution of the absolute value of the maximum
        # in the signal-plus-noise model, t_n is sqrt(n)
        if boot_data_type == 'obs':
            dist = np.sqrt(nsubj) * (np.abs(boot_means -
                                            sample_mean.repeat(m_boots).reshape(boot_stat_dim)) / sigma).max(
                tuple(np.arange(d)))
        elif boot_data_type == 'res':
            dist = np.sqrt(nsubj) * np.abs(boot_means / sigma).max(
                tuple(np.arange(d)))
    elif boot_type in ['multiplier']:
        # multiplier: array of shape (n_subj, m_boots), for multiplier bootstrap
        if multiplier == "g":
            multiplier = np.random.normal(size=(nsubj, m_boots))
        elif multiplier == "r":
            multiplier = np.random.choice([1, -1], size=(nsubj, m_boots), replace=True)
        # get bootstrap mean
        boot_means = (concat_data @ multiplier).reshape(boot_stat_dim) / nsubj
        # get sample or bootstrap standard deviation sigma
        if standardize == 'z':
            sigma = data.std(axis=-1, ddof=1).repeat(m_boots).reshape(boot_stat_dim)
        elif standardize == 't':
            boot_mean_squared = (concat_data ** 2 @ multiplier ** 2).reshape(boot_stat_dim) / nsubj
            sigma = np.sqrt(np.abs(boot_mean_squared - boot_means ** 2) * (nsubj / (nsubj - 1)))
        # compute bootstrap distribution of the absolute value of the maximum
        if boot_data_type == 'obs':
            dist = np.sqrt(nsubj) * (np.abs(
                boot_means - sample_mean.repeat(m_boots).reshape(boot_stat_dim)) / sigma).max(
                tuple(np.arange(d)))
        elif boot_data_type == 'res':
            dist = np.sqrt(nsubj) * np.abs(boot_means / sigma).max(
                tuple(np.arange(d)))
    q = np.quantile(dist, 1 - alpha)
    # print('sigma:',sigma)
    return dist, q


def confband(data, alpha=0.05, m_boots=5000,
             boot_data_type='res', boot_type='multiplier',
             standardize='t', multiplier='r', print_q=False):
    """
    Compute simultaneous confidence band(SCB) for the mean of a sample from a functional signal plus noise model

    Parameters
    ----------
    data: array(int or float)
      array of dimension K_1 x ... x K_D x N
      containing N-realizations of a random field over a D-dimensional domain
    alpha: float, optional, default: 0.05
      1-alpha is the targeted covering rate of the simultaneous confidence interval
    m_boots: int, optional, default: 5e3
      number of bootstrap samples
    boot_data_type: string, optional, default: "res"
      the type of the input data.
      Options are "res" for residuals, "obs" for observations.
    boot_type: string, optional, default: "multiplier"
      the type of bootstrap
      Options are "multiplier" for multiplier bootstrap and "nonparametric" for nonparametric bootstrap.
    standardize: string, optional, default: "t"
      the type of standardization
      Options are "t" for t score(use bootstrap sd) and "z" for z score(only use sample sd).
    multiplier: string, optional, default: "r"
      the type of multiplier for multiplier bootstrap, only used when boot_type="multiplier"
      Options are "r" for Rademacher multipliers and "g" for Gaussian multipliers.
    print_q: bool

    Returns
    -------
    tuple with elements:
    est: array
      estimate
    lower: array
      lower limits of the SCB
    upper: array
      upper limits of the SCB

    Details
    -------
    A particular bootstrap method can be specified based the following three aspects:
      1. what to bootstrap on:
         residuals(boot_data_type='res') or original observations(boot_data_type='obs').
      2. how to generate bootstrap samples:
         resample with replacement(ie, nonparametric bootstrap, boot_type='nonparametric') or
         use multipliers (ie, multiplier bootstrap, boot_type='multiplier')
      3. how to standardize:
         t score(use bootstrap sd, standardize= t') or z score(only use sample sd, standardize='z')

    For example:
    - the non-parametric bootstrap in Degras (2011)
      "Simultaneous confidence bands for non-parametric regression with functional data" can be specified with:
        boot_data_type='obs', boot_type='nonparametric', standardize='z'
    - the multiplier bootstrap in Cherno can be specified with:
        boot_data_type='res', boot_type='multiplier', standardize = 'z', multiplier='r'(or 'g'),
    - the multiplier-t bootstrap in Telschow Schwartzmann (2019)
      "Simultaneous confidence bands for non-parametric regression with functional data" can be specified with:
        boot_data_type='res', boot_type='multiplier', standardize = 't', multiplier = 'r'(or 'g')

    Usually, holding other aspects the same, the z- and the t-standardization agree for large sample sizes.
    However, for small sample sizes the t-standardization has better covering rates,
    but higher variability in the estimate of the quantile.


    Example
    -------
    data = np.random.normal(size=(30,40,50))
    confband(data)
    confband(data, boot_data_type='obs', boot_type='nonparametric')
    """

    # -----check user inputs
    if not isinstance(m_boots, int):
        sys.exit("The input m_boots needs to be a positive natural number.")

    # -----computation
    data_dim = data.shape
    nsubj = data_dim[-1]
    sample_mean = data.mean(axis=-1)
    sigma = data.std(axis=-1, ddof=1)
    se = sigma / np.sqrt(nsubj)

    if boot_data_type == 'res':
        # use residuals instead of original observations
        data = data - sample_mean.repeat(nsubj).reshape(data_dim)

    _, q = bootstrap(data, alpha=alpha, m_boots=m_boots, boot_data_type=boot_data_type, boot_type=boot_type,
                     standardize=standardize, multiplier=multiplier)

    est = sample_mean
    lower = est - q * se
    upper = est + q * se

    if print_q:
        if boot_type == 'multiplier':
            print(f'{alpha}th upper quantile of the distribution of the maximum of the random field:{q:.3f},'
                  f'computed from {boot_type} bootstrap with {boot_data_type}, {standardize} standardization and {multiplier} multiplier')
        if boot_type == 'nonparametric':
            print(f'{alpha}th upper quantile of the distribution of the maximum of the random field:{q:.3f},'
                  f'computed from {boot_type} bootstrap with {boot_data_type}, {standardize} standardization')
        """
        if boot_type == 'multiplier': 
            multiplier = 'NA'

        q_df = pd.DataFrame({'q': q,
                      'n': nsubj, 'data_dim':"x".join([str(x) for x in data_dim[:-1]]),
                      'alpha': alpha, 'm_boots': m_boots,
                      'boot_data_type': boot_data_type, 
                      'boot_type': boot_type, 'standardize': standardize, 'multiplier': multiplier},
                     index = [0])    
        return est, lower, upper, q_df
        """
    return est, lower, upper


