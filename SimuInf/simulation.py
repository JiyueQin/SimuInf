import numpy as np
import pandas as pd
import sys
import time
from SimuInf.scb import confband
from SimuInf.random_field_generator import gen_2D


# to do: update
def scb_cover_rate(dim=None, shape=None, shape_spec=None, noise_type='gaussian', data_sim=None, mu=None,
                   m_sim=1000, alpha=0.05, m_boots=5000, boot_data_type='res', boot_type='multiplier',
                   standardize='t', multiplier='r', std=None):
    """
    Calculate the covering rate of SCB constructed by various methods.
    Parameters
    ----------
    data_sim: list, optional, default: None
      a list of arrays as the generated data for simulation. If None, random 2D fields will be generated based on other parameters supplied.
      The length of the list should be the same as m_sim.
    mu: array, optiona;, default: None
      an array for the mean of the data. If None, this will be the mean of random 2D fields generated based on other parameters supplied.
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

    Returns
    -------
    a pandas dataframe with one row

    Details
    -------


    Example
    -------


    """
    # -----check user inputs
    if not isinstance(m_boots, int):
        sys.exit("The input m_boots needs to be a positive natural number.")

    # -----simulation
    # create simulated data if not provided: a list of arrays
    if data_sim is None:
        print('----creating simulated data----')
        supply_data = False
        data_sim = []
        for i in range(m_sim):
            ## generate 2D data
            if shape == 'noise':
                data = np.random.normal(size=dim) * std
                mu = np.zeros((dim[0], dim[1]))
            else:
                # to do: update gen_2D, to fix the dim issue, also need to fix funcations of ellipse_2d..
                data, mu = gen_2D(dim=(dim[2], dim[0], dim[1]), shape=shape, shape_spec=shape_spec,
                                  noise_type=noise_type)
                data_sim.append(data)
    else:
        supply_data = True
        # if m_data_sim != m_sim:
        #     sys.exit("The length of the supplied data should be the same as m_sim.")
        print(f'-----use provided data and mu, the simulation replication number is {len(data_sim)} ---- ')
        dim = data_sim[0].shape
    # apply the method on the simulated data
    cover_ls = []
    q_ls = []
    start_time = time.time()
    for data in data_sim:
        ## compute SCB
        if not supply_data:
            data = np.moveaxis(data, 0, -1)
        est, lower, upper, q = confband(data, alpha, m_boots, boot_data_type, boot_type,
                                        standardize, multiplier, return_q=True)
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
        q_ls.append(q)
    # compute the TOTAL runtime for all runs of the method, divide by m_sim gives the runtime for each run
    runtime_secs = round((time.time() - start_time), 2)
    # ?width length comfirm
    if boot_type in ['nonparametric']:
        multiplier = 'NA'
    if not supply_data:
        out = pd.DataFrame({'rate': np.mean(cover_ls),
                            'mean_q': np.mean(q_ls),
                            'sd_q': np.std(q_ls, ddof=1),
                            'runtime_secs': runtime_secs,
                            'n': dim[-1], 'w': dim[0], 'h': dim[1],
                            'shape': shape, 'fwhm_noise': shape_spec['fwhm_noise'],
                            'fwhm_signal': shape_spec['fwhm_signal'],
                            'std': shape_spec['std'],
                            'noise_type': noise_type,
                            'alpha': alpha, 'm_boots': m_boots,
                            'boot_data_type': boot_data_type,
                            'boot_type': boot_type, 'standardize': standardize, 'multiplier': multiplier},
                           index=[0])
    else:
        out = pd.DataFrame({'rate': np.mean(cover_ls),
                            'mean_q': np.mean(q_ls),
                            'sd_q': np.std(q_ls, ddof=1),
                            'runtime_secs': runtime_secs,
                            'n': dim[-1], 'dim': str(dim[:-1]),
                            'alpha': alpha, 'm_boots': m_boots,
                            'boot_data_type': boot_data_type,
                            'boot_type': boot_type, 'standardize': standardize, 'multiplier': multiplier},
                           index=[0])
    return out


# to do : update
def scb_cover_rate_df(setting_df, method_df,
                      m_sim=1000, alpha=0.05,
                      m_boots=5000):
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
    for i in setting_df.index:
        print(
            f'----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----')
        dim = (setting_df['w'][i], setting_df['h'][i], setting_df['n'][i])
        # print(dim)
        shape = setting_df['shape'][i]
        # print(shape)
        # gen_spec returns a tuple (specs for 50*50, specs for 100*100)
        shape_spec_ls = \
        gen_spec(fwhm_sig=10, fwhm_noise=setting_df['fwhm_noise'][i], std=setting_df['std'][i], mag=4, r=0.5)[0]

        if shape == 'circular':
            shape_spec = shape_spec_ls[0]
        elif shape == 'elipse':
            shape_spec = shape_spec_ls[1]
        elif shape == 'ramp':
            shape_spec = shape_spec_ls[2]
            # print(shape_spec)

        #  create simulated data: a list of arrays
        data_sim = []
        for j in range(m_sim):
            ## generate 2D data
            if shape == 'noise':
                data = np.random.normal(size=(dim[2], dim[0], dim[1])) * std
                mu = np.zeros((dim[0], dim[1]))
            else:
                # to do: update gen_2D
                data, mu = gen_2D(dim=(dim[2], dim[0], dim[1]), shape=shape, shape_spec=shape_spec,
                                  noise_type=setting_df['noise_type'][i])
                data_sim.append(data)
        # apply each method to the simulated data
        for k in method_df.index:
            # todo: need to update here since scb_cover_rate is changed
            df_single = scb_cover_rate(dim, shape, shape_spec, noise_type=setting_df['noise_type'][i],
                                       data_sim=data_sim, mu=mu,
                                       m_sim=m_sim, alpha=alpha, m_boots=m_boots,
                                       boot_data_type=method_df['boot_data_type'][k],
                                       boot_type=method_df['boot_type'][k],
                                       standardize=method_df['standardize'][k],
                                       multiplier=method_df['multiplier'][k])
            df = pd.concat([df, df_single], ignore_index=True)

    return df


