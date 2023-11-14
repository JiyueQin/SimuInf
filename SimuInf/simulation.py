def scb_cover_rate(dim, shape, shape_spec=None,
                   m_sim=1000, alpha = 0.05,
                   m_boots = 5000, method = 'multi_t', multiplier = 'r', std=None):
  """
  Calculate the covering rate of SCB constructed by various methods.

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
  in the estimate of the quantile than the simple non-paramtric bootstrap.

  Example
  -------
  data = np.random.normal(size=(30,40,50))
  bootstrap(data)
  bootstrap(data, m_boots=100, method='multi_regular', multiplier = 'g')

  """
  start_time = time.time()
  #-----check user inputs
  if not isinstance(m_boots, int):
    sys.exit("The input m_boots needs to be a positive natural number.")

  #-----simulation
  cover_ls = []
  for i in range(m_sim):
    ## generate 2D data
    if shape == 'noise':
      data = np.random.normal(size=dim) * std
      mu = np.zeros((dim[0], dim[1]))

    else:
      # to do: update gen_2D
      data, mu = gen_2D(dim=(dim[2], dim[0], dim[1]), shape=shape, shape_spec=shape_spec)

    ## compute SCB
    data = np.moveaxis(data, 0,-1)
    est, lower, upper = confband(data, alpha,  m_boots, method, multiplier)
    ## evaluate coverage
    cover = np.all(lower <= mu) and np.all(mu <= upper)
    """
    # checking
    fig, axs = plt.subplots(1,4,figsize=(10, 6))
    for i, (arr, name) in enumerate(zip([mu, est, lower, upper],
                                ['mu','est','lower','upper'])):

      im=axs[i].imshow(arr)
      axs[i].set_title(name, fontsize = 10)

    cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    print('mu', mu)
    print('est', est)
    print('lower', lower)
    print('upper', upper)
    print('number of locations with lower<=mu:', np.sum(lower <= mu))
    print('number of locations with mu<=upper:', np.sum(mu <= upper))
    print('number of locations with coverage:', np.logical_and(lower <= mu,mu <= upper).sum())
    """
    cover_ls.append(cover)

  runtime_mins = round((time.time() - start_time)/60,2)
  #?width length comfirm
  if method in ['regular', 't']:
    multiplier = 'NA'
  out = pd.DataFrame({'rate': np.mean(cover_ls),
                      'runtime_mins': runtime_mins,
                      'n': dim[-1], 'w':dim[0], 'h':dim[1],
                      'shape': shape, 'fwhm_noise':shape_spec['fwhm_noise'],
                      'fwhm_signal': shape_spec['fwhm_signal'],
                      'alpha': alpha, 'm_boots': m_boots,
                      'method': method, 'multiplier': multiplier},
                     index = [0])
  return out



def scb_cover_rate_df(dim, shape_ls, shape_spec_ls,
                   m_sim=1000, alpha = 0.05,
                   m_boots = 5000, method_ls = ['t', 'regular', 'multi_t', 'multi_regular'],
                   multiplier_ls = ['r','g']):
  """
  Calculate the covering rate of SCB constructed by various methods in multiple experiments.

  Parameters
  ----------
  data: array(int or float)
    array of dimension K_1 x ... x K_D x N containing N-realizations of a random field over a D-dimensional domain.
  alpha: float, optional, default: 0.05
    1-alpha is the targeted covering rate of the simutaneous confidence interval
  m_boots: int, optional, default: 5e3
    number of bootstrap samples
  method: string, optional, default: "multi_t"
    the method of bootstrap. Options are "t", "regular", "multi_regular", "multi_t", see details.
  multiplier: string, optional, default: "r"
    multipliers for multiplier bootstrap, only used when method is "multi_regular" or "multi_t".
    Options are "r" for rademacher multiplers and "g" for Gaussian multipliers.

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
  in the estimate of the quantile than the simple non-paramtric bootstrap.

  Example
  -------
  data = np.random.normal(size=(30,40,50))
  bootstrap(data)
  bootstrap(data, m_boots=100, method='multi_regular', multiplier = 'g')

  """
  df = pd.DataFrame()
  for i, shape in enumerate(shape_ls):
    shape_spec = shape_spec_ls[i]
    for method in method_ls:
      if method in ['regular', 't']:
        df_single = scb_cover_rate(dim, shape, shape_spec,
                   m_sim, alpha,
                   m_boots, method)
        df = pd.concat([df, df_single])
      elif method in ['multi_regular', 'multi_t']:
        for multiplier in multiplier_ls:
          df_single = scb_cover_rate(dim, shape, shape_spec,
                   m_sim, alpha,
                   m_boots, method, multiplier)
          df = pd.concat([df, df_single], ignore_index=True)
  return df

