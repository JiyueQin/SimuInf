import numpy as np
from scipy.ndimage import gaussian_filter
import sys

def gen_spec(fwhm_sig=10, fwhm_noise=5, std=5, mag=4, r=0.5):
  """
  Generate dictionaries of the shape spec sets to be used in random field generator.

  Parameters
  ----------
  fwhm_sig : int
    full width at half maximum for signal
  fwhm_noise : int
    full width at half maximum for noise field
  std : int
    standard deviation for the noise field N(0, std^2)
  mag : int
    magnitude of the signal
  r : int
    radii of ellipses

  Returns
  -------
  spec_dic_set_50 : list
    list of three dictionaries (circular, ellipse, ramp) for image dimension 50*50
  spec_dic_set_100 : list
    list of three dictionaries (circular, ellipse, ramp) for image dimension 100*100

  Examples
  --------
  spec_50, spec_100 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=5, mag=4, r=0.5)
  gen_2D(80, 50, 50)), shape="ellipse", shape_spec=spec_50[0])
  """
  cir_50 = {'a':r, 'b':r, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':fwhm_sig/2}
  elp_50 = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':fwhm_sig/2}
  ramp_50 = {'direction':1, 'std':std,'mag':(0,mag), 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':0}
  spec_dic_set_50 = [cir_50, elp_50, ramp_50]

  cir_100 = {'a':r, 'b':r, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_sig}
  elp_100 = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_sig}
  ramp_100 = {'direction':1, 'std':std,'mag':(0,mag), 'fwhm_noise':fwhm_noise, 'fwhm_signal':0}
  spec_dic_set_100 = [cir_100, elp_100, ramp_100]

  return(spec_dic_set_50, spec_dic_set_100)


def ramp_2D(dim, shape_spec):
  """
  Generate a 2D ramp signal.

  Parameters
  ----------
  dim : int
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters including direction, mag, and std

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = ramp_2D(dim=dim, shape_spec=shape_spec)
  """
  nsubj = dim[0]
  direction = shape_spec['direction']
  mag = shape_spec['mag']
  std = shape_spec['std']

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  if direction == 1: #horizontal
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2]),dim[1]).reshape(dim[2],dim[1]).transpose()
  mu = np.array(mu_temp, dtype='float')
  return(mu)

def ellipse_2D(dim, shape_spec, truncate=4):
  """
  Generate a 2D ellipse signal.

  Parameters
  ----------
  dim : int
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters including a, b, mag, std, and fwhm_signal
  truncate : int
    truncation point for the smoothing

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  """
  nsubj = dim[0]
  a = shape_spec['a']
  b = shape_spec['b']
  mag = shape_spec['mag']
  fwhm_signal = shape_spec['fwhm_signal']

  # signal
  x, y = np.meshgrid(np.linspace(-1,1,dim[1]), np.linspace(-1,1,dim[2]))
  cx, cy = 0,0
  theta = -np.pi/4
  xx = np.cos(theta)*(x-cx) + np.sin(theta)*(y-cy)
  yy = -np.sin(theta)*(x-cx) + np.cos(theta)*(y-cy)
  ellipse = np.array((xx/a)**2 + (yy/b)**2 <= 1, dtype="float")

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  ellipse_smth = gaussian_filter(ellipse, sigma = sigma_signal, truncate=truncate)
  mu = np.array(ellipse_smth * mag, dtype='float')

  return(mu)


def gen_2D(dim, shape, shape_spec, truncate=4, noise_type='gaussian', noise_df=None):
    """
    Generate a 2D random field using white noise smoothed with Gaussian kernel.

    Parameters
    ----------
    dim : int
      dimension of the image (N, W, H)
    shape : str
      shape of the signal; choose from ramp or ellipse which includes circular
    truncate : int, optional, default: 4
      truncate the filter at this many standard deviations.
    noise_type: string, optional, default: gaussian.
      the distribution of the noise. Options are "gaussian" (for standard gaussian) and "t" (for t distribution).
    noise_df: int, optional, default: 3
      the degree of freedom of the noise distribution if it has a t distribution. This parameter is only used if noise_type='t'.

    Returns
    -------
    data : array
      generated 2D field
    mu : array
      generated 2D signal (mu) of the random field

    Examples
    --------
    spec_50, spec_100 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=5, mag=4, r=0.5)
    gen_2D((80, 50, 50), shape="ellipse", shape_spec=spec_50[0])
    """
    fwhm_noise = shape_spec['fwhm_noise']
    std = shape_spec['std']
    nsubj = dim[0]

    # signal
    if shape == "ramp":
        mu = ramp_2D(dim=dim, shape_spec=shape_spec)
    else:
        mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)

    # noise
    # fwmh = sqrt(8log2)*sigma
    sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))
    # start with a 2D field of larger dimension for later cropping to the specified dimension
    padding = round(truncate * sigma_noise)
    dim_larger = [dim[0], dim[1] + 2 * padding, dim[2] + 2 * padding]
    if noise_type == 'gaussian':
        noise_raw = np.random.randn(*dim_larger)
        sd_before_smoothing = 1
    elif noise_type == 't':
        if noise_df is None:
            noise_df = 3
        noise_raw = np.random.standard_t(noise_df, dim_larger)
        if noise_df > 2:
            sd_before_smoothing = np.sqrt(noise_df / (noise_df - 2))
        else: 
            sys.exit("Noise degree of freedom needs to be >2 for a t distribution!")
    elif noise_type == 'chisq':
        if noise_df is None:
            noise_df = 5
        # center by subtracting the mean
        noise_raw = np.random.chisquare(noise_df, dim_larger)-noise_df
        sd_before_smoothing = np.sqrt(2*noise_df)
    #noise_raw_sd = noise_raw.std(0, ddof=1)
    #print('SD before smoothing across all subjects for each pixel,', 'mean:', noise_raw_sd.mean(), 'max:', noise_raw_sd.max(), 'min:', noise_raw_sd.min())
    # smoothing the noise with gaussian kernel
    for i in np.arange(nsubj):
        noise_raw[i, :, :] = gaussian_filter(noise_raw[i, :, :], sigma=sigma_noise, truncate=truncate)
    # rescale the smoothed noise to have sd of the specified sd: std
    # method1: shortcut
    # noise = noise / np.mean(np.std(noise, 0, ddof=1)) * std
    # method2: more accurate
    # calculate the sd of the 2d field after smoothing for pixels in the center
    sig = np.zeros((dim[1], dim[2]))
    sig[int(dim[1] / 2), int(dim[2] / 2)] = 1
    # retrieved the Gaussian kernel used for smoothing
    kernel = gaussian_filter(sig, sigma=sigma_noise, truncate=truncate)
    scale = np.sqrt(np.sum(kernel ** 2))  # scale the smoothed noise field by sqrt(sum(kernel^2))
    sd_after_smoothing = scale * sd_before_smoothing
    noise = noise_raw[:, padding:padding + dim[1], padding:padding + dim[2]]
    noise = noise / sd_after_smoothing * std
    #noise_sd = noise.std(0, ddof=1)
    #print('SD after smoothing across all subjects for each pixel,', 'mean:', noise_sd.mean(), 'max:', noise_sd.max(), 'min:', noise_sd.min(), 'sd:', noise_sd.std())
    data = np.array(mu + noise, dtype='float')
    return data, mu


