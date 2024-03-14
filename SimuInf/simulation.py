import numpy as np
import pandas as pd
import sys
import time
from SimuInf.scb import confband
from SimuInf.confset import confset
from SimuInf.random_field_generator import gen_2D
from confidenceset.random_field_generator import gen_spec


def scb_coverage(data, mu, alpha=0.05, m_boots=5000,
                boot_data_type='res', boot_type='multiplier',
                standardize='t', multiplier='r', thresholds_ls = None):
    """
    Check the coverage of SCB by a specific method with given data and mean.
    Parameters
    ----------
    mu: array, optional;, default: None
      an array for the mean of the data. If None, this will all 0.
    alpha: float, optional, default: 0.05
      1-alpha is the targeted covering rate of the simultaneous confidence interval
    m_boots: int, optional, default: 5e3
      number of bootstrap samples

    Returns
    -------
    tuple with elements:
    cover: int
      1 indicates coverage of true mean and 0 otherwise
    q: float
      alpha-quantile value of dist
    runtime_secs: float
      runtime

    Details
    -------


    Example
    -------


    """
    start_time = time.time()
    est, lower, upper, q = confband(data, alpha, m_boots, boot_data_type, boot_type,
                                     standardize, multiplier, return_q=True)
    # evaluate coverage
    cover = np.all(lower <= mu) and np.all(mu <= upper)
    # compute the runtime for each run of the method
    runtime_secs = time.time() - start_time
    if thresholds_ls is not None:
        df = pd.DataFrame()
        for i in range(len(thresholds_ls)):
            thresholds = thresholds_ls[i]
            n_thresholds = len(thresholds)
            for threshold in thresholds:
                true_set = mu>threshold
                est_set, inner_set, outer_set = confset(est, lower, upper, threshold)
                cover_set_single = np.all(true_set<=outer_set) and  np.all(true_set>=inner_set)
                if cover_set_single == 0:
                    break
            cover_set = cover_set_single
            df_single = pd.DataFrame({'thresholds_index': i, 'n_thresholds': n_thresholds,
                                      'cover_set': cover_set},
                                     index=[0])
            df = pd.concat([df, df_single], ignore_index=True)
        return df
    else:
        df = pd.DataFrame({'cover': cover,
                           'q': q,
                           'runtime_secs': runtime_secs},
                          index=[0])
        return df


def scb_cover_rate(method_df, dim=None, shape=None, shape_spec=None, noise_type='gaussian',
                   data_in=None, mu=None, subsample_size=20,
                   m_sim=1000, alpha=0.05, m_boots=5000, std=None, thresholds_ls = None):
    """
    Calculate the covering rate of SCB constructed by various methods.
    Parameters
    ----------
    method_df: DataFrame,
      a dataframe where each row is a particular method
    mu: array, optional;, default: None
      an array for the mean of the data. If None, this will all 0.
    alpha: float, optional, default: 0.05
      1-alpha is the targeted covering rate of the simultaneous confidence interval
    m_boots: int, optional, default: 5e3
      number of bootstrap samples

    Returns
    -------
    a pandas dataframe where each row is the coverage of each method

    Details
    -------


    Example
    -------


    """
    # -----check user inputs
    if not isinstance(m_boots, int):
        sys.exit("The input m_boots needs to be a positive natural number.")
    subsampling = (data_in is not None)
    provide_thresholds = (thresholds_ls is not None)
    if subsampling:
        dim = data_in.shape
    # initialize mu to be all 0
    if mu is None:
        # note for 1d, this becomes a row vector.  to do: fix the issue of input mu is a column vector
        mu = np.zeros(dim[:-1])

    # reset the index
    method_df = method_df.reset_index(drop=True)
    df = pd.DataFrame()
    # -----simulation

    for i in range(m_sim):
        # perform subsampling
        if subsampling:
            # get a subsample by sampling without replacement
            data = data_in[..., np.random.choice(dim[-1], subsample_size, replace=False)]
        # simulate 2D data
        elif shape == 'noise':
            data = np.random.normal(size=dim) * std
        else:
            # to do: update gen_2D, to fix the dim issue, also need to fix functions of ellipse_2d.
            data, mu = gen_2D(dim=(dim[2], dim[0], dim[1]), shape=shape, shape_spec=shape_spec, noise_type=noise_type)
            data = np.moveaxis(data, 0, -1)
        for k in method_df.index:
            # apply the method on the simulated data
            if provide_thresholds:
                df_single = scb_coverage(data, mu, alpha, m_boots,
                                          boot_data_type=method_df['boot_data_type'][k],
                                          boot_type=method_df['boot_type'][k],
                                          standardize=method_df['standardize'][k],
                                          multiplier=method_df['multiplier'][k], thresholds_ls = thresholds_ls)
            else:
                df_single = scb_coverage(data, mu, alpha, m_boots,
                                          boot_data_type=method_df['boot_data_type'][k],
                                          boot_type=method_df['boot_type'][k],
                                          standardize=method_df['standardize'][k],
                                          multiplier=method_df['multiplier'][k])
            df_single =df_single.assign(simu_index=i, method_index=k)
            df = pd.concat([df, df_single], ignore_index=True)
    if provide_thresholds:
        df_summary = df.groupby(['method_index', 'thresholds_index', 'n_thresholds']).agg({'cover_set': 'mean'})
        df_summary.columns = ['rate']
        df_summary = df_summary.reset_index()
    else:
        df_summary = df.groupby('method_index').agg({'cover': 'mean', 'q': ['mean', 'std'], 'runtime_secs': 'mean'})
        df_summary.columns = ['rate', 'mean_q', 'sd_q', 'runtime_secs']
    df_summary = df_summary.join(method_df, on='method_index')

    # ?width length confirm
    if subsampling:
        out = df_summary.assign(n=dim[-1], subsample_size=subsample_size, dim=str(dim[:-1]),
                                alpha=alpha, m_boots=m_boots, m_sim=m_sim)
    else:
        out = df_summary.assign(n=dim[-1], w=dim[0], h=dim[1],
                                shape=shape, fwhm_noise=shape_spec['fwhm_noise'],
                                fwhm_signal=shape_spec['fwhm_signal'],
                                std=shape_spec['std'],
                                noise_type=noise_type, alpha=alpha, m_boots=m_boots, m_sim=m_sim)
    return out


def scb_cover_rate_multiple(setting_df, method_df,
                            m_sim=1000, alpha=0.05,
                            m_boots=5000, data_in=None, mu=None, thresholds_ls = None):
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
    a pandas dataframe

    Details
    -------


    Example
    -------

    """
    df = pd.DataFrame()
    # reset the index
    setting_df = setting_df.reset_index(drop=True)
    method_df = method_df.reset_index(drop=True)
    subsampling = (data_in is not None)
    for i in setting_df.index:
        print(
            f'----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----')
        if not subsampling:
            dim = (setting_df['w'][i], setting_df['h'][i], setting_df['n'][i])
            shape = setting_df['shape'][i]

            # gen_spec returns a tuple (specs for 50*50, specs for 100*100)
            shape_spec_ls = gen_spec(fwhm_sig=10, fwhm_noise=setting_df['fwhm_noise'][i],
                                     std=setting_df['std'][i], mag=4, r=0.5)[0]

            if shape == 'circular':
                shape_spec = shape_spec_ls[0]
            elif shape == 'ellipse':
                shape_spec = shape_spec_ls[1]
            elif shape == 'ramp':
                shape_spec = shape_spec_ls[2]

        if subsampling:
            df_single = scb_cover_rate(method_df, data_in=data_in, mu=mu,
                                       subsample_size=setting_df['subsample_size'][i],
                                       m_sim=m_sim, alpha=alpha, m_boots=m_boots, thresholds_ls=thresholds_ls)
        else:
            df_single = scb_cover_rate(method_df, dim=dim, shape=shape, shape_spec=shape_spec,
                                       noise_type=setting_df['noise_type'][i],
                                       m_sim=m_sim, alpha=alpha, m_boots=m_boots, thresholds_ls=thresholds_ls)

        df = pd.concat([df, df_single], ignore_index=True)

    return df




