import numpy as np
import sys
def bootstrap(data, alpha = 0.05, m_boots = 5000,
              method = 'multi_t', multiplier = 'r'):
  """
  Compute bootstrap estimator for the quantile of the maximum of a random field.

  Parameters
  ----------
  data: array(int or float)
    array of dimension K_1 x ... x K_D x N containing N-realizations of a random field over a D-dimensional domain.
  alpha: float, optional, default: 0.05
    1-alpha is the targeted covering rate of the simultaneous confidence interval
  m_boots: int, optional, default: 5e3
    number of bootstrap samples
  method: string, optional, default: "multi_t"
    the method of bootstrap. Options are "t", "regular", "multi_regular", "multi_t", see details.
  multiplier: string, optional, default: "r"
    multipliers for multiplier bootstrap, only used when method is "multi_regular" or "multi_t".
    Options are "r" for rademacher multipliers and "g" for Gaussian multipliers.

  Returns
  -------
  tuple with elements:
  dist: list
    realizations of maximum of the random field in bootstrap
  q: float
    alpha-quantile value of dist

  Details
  -------
  Four methods are available:
  "regular": a non-parametric bootstrap.
  "t": a bootstrap-t version based on Degras (2011)
       "Simultaneous confidence bands for non-parametric regression with functional data"
  "multi_regular":  multiplier-regular version based on Cherno
  "multi_t": multiplier-t version based on Telschow Schwartzmann (2019)
       "Simultaneous confidence bands for non-parametric regression with functional data"

  Note, with "regular" and "t", the input data should be original data,
  while with "multi_regular" and "multi_t", the input data should be residuals and there are multiple ways to calculate residuals.

  For large sample sizes the regular and the t versions do agree, however,
  for small sample sizes the bootstrap-t has better covering rates, but higher variability
  in the estimate of the quantile than the simple non-parametric bootstrap.

  Example
  -------
  data = np.random.normal(size=(30,40,50))
  bootstrap(data)
  bootstrap(data, m_boots=100, method='multi_regular', multiplier = 'g')

  """

  #-----check user inputs
  if not isinstance(m_boots, int):
    sys.exit("The input m_boots needs to be a positive natural number.")

  #-----computation
  data_dim = data.shape
  nsubj = data_dim[-1]
  D = len(data_dim) - 1
  boot_stat_dim = (*data_dim[:-1], m_boots)
  concat_data = data.reshape((np.prod(data_dim[:-1]), nsubj))
  if method in ['regular', 't']:
    # counter: array of shape (n_subj, m_boots), for bootstrap resampling
    counter = np.random.multinomial(nsubj, np.repeat(1/nsubj, nsubj),
                                  m_boots).transpose()
    # get sample mean and bootstrap mean
    sample_mean = data.mean(axis = -1)
    boot_means = (concat_data @ counter).reshape(boot_stat_dim)/nsubj

    # get sample or bootstrap standard deviation sigma
    if method == 'regular':
      sigma = data.std(axis = -1, ddof = 1).repeat(m_boots).reshape(boot_stat_dim)
    elif method == 't':
      boot_mean_squared = (concat_data**2 @ counter).reshape(boot_stat_dim)/nsubj
      sigma = np.sqrt(np.abs(boot_mean_squared - boot_means**2) * (nsubj/(nsubj-1)))
    # compute bootstrap distribution of the absolute value of the maximum
    ## in the signal-plus-noise model, t_n is sqrt(n)
    dist = np.sqrt(nsubj) * (np.abs(
        boot_means-sample_mean.repeat(m_boots).reshape(boot_stat_dim))/sigma).max(
            tuple(np.arange(D)))
  elif method in ['multi_regular', 'multi_t']:
    # multiplier: array of shape (n_subj, m_boots), for multiplier bootstrap
    if multiplier == "g":
        multiplier = np.random.normal(size=(nsubj, m_boots))
    elif multiplier == "r":
        multiplier = np.random.choice([1, -1], size=(nsubj, m_boots), replace=True)
    # get bootstrap mean
    boot_means = (concat_data @ multiplier).reshape(boot_stat_dim)/nsubj
    # get sample or bootstrap standard deviation sigma
    if method == 'multi_regular':
      sigma = data.std(axis = -1, ddof = 1).repeat(m_boots).reshape(boot_stat_dim)
    elif method == 'multi_t':
      boot_mean_squared = (concat_data**2 @ multiplier ** 2).reshape(boot_stat_dim)/nsubj
      sigma = np.sqrt(np.abs(boot_mean_squared - boot_means ** 2) * (nsubj/(nsubj-1)))
    # compute bootstrap distribution of the absolute value of the maximum
    dist = np.sqrt(nsubj) * np.abs(boot_means/sigma).max(
            tuple(np.arange(D)))
  q = np.quantile(dist, 1-alpha)
  #print('sigma:',sigma)
  return (dist, q)





def confband(data, alpha = 0.05,  m_boots = 5000,
              method = 'multi_t', multiplier = 'r', print_q=False):
  """
  Compute simultaneous confidence band(SCB) for the mean of a sample from a functional signal plus noise model

  Parameters
  ----------
  data: array(int or float)
    array of dimension K_1 x ... x K_D x N containing N-realizations of a random field over a D-dimensional domain.
  alpha: float, optional, default: 0.05
    1-alpha is the targeted covering rate of the simutaneous confidence interval
  m_boots: int, optional, default: 5e3
    number of bootstrap samples
  method: string, optional, default: "multi_t"
    the method to estimate the quantile to construct SCB. Options are "t", "regular", "multi_regular", "multi_t", see details.
  multiplier: string, optional, default: "r"
    multipliers for multiplier bootstrap, only used when method is "multi_regular" or "multi_t".
    Options are "r" for rademacher multipliers and "g" for Gaussian multipliers.

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
  Four methods are available:
  "regular": a non-parametric bootstrap.
  "t": a bootstrap-t version based on Degras (2011)
       "Simultaneous confidence bands for non-parametric regression with functional data"
  "multi_regular":  multiplier-regular version based on Cherno
  "multi_t": multiplier-t version based on Telschow Schwartzmann (2019)
       "Simultaneous confidence bands for non-parametric regression with functional data"

  For large sample sizes the regular and the t versions do agree, however,
  for small sample sizes the bootstrap-t has better covering rates, but higher variability
  in the estimate of the quantile than the simple non-paramtric bootstrap.

  Example
  -------
  data = np.random.normal(size=(30,40,50))
  confband(data)
  confband(data, m_boots=100, method='multi_regular', multiplier = 'g')
  """

  #-----check user inputs
  if not isinstance(m_boots, int):
    sys.exit("The input m_boots needs to be a positive natural number.")

  #-----computation
  data_dim = data.shape
  nsubj = data_dim[-1]
  sample_mean = data.mean(axis = -1)
  if method in ['regular', 't']:
    _, q = bootstrap(data, alpha = alpha, m_boots=m_boots, method=method)
  elif method in ['multi_regular', 'multi_t']:
    ## use residuals instead of original data for multiplier bootstrap
    residuals = data - sample_mean.repeat(nsubj).reshape(data_dim)
    _, q = bootstrap(residuals, alpha = alpha, m_boots=m_boots,
                     method=method, multiplier = multiplier)

  sigma = data.std(axis = -1, ddof = 1)
  se = sigma/np.sqrt(nsubj)
  if print_q:
    print(f'{alpha}th upper quantile of the distribution of the maximum of the random field:{q:.3f}')

  est = sample_mean
  lower = est - q*se
  upper = est + q*se
  return (est, lower, upper)





