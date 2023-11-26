import numpy as np
from scipy.ndimage import gaussian_filter
from confidenceset.random_field_generator import ramp_2D, ellipse_2D


def gen_2D(dim, shape, shape_spec, truncate=4):
    """
    generates a 2D random field using white noise smoothed with Gaussian kernel.

    Parameters
    ----------
    dim : int
      dimension of the image (N, W, H)
    shape : str
      shape of the signal; choose from ramp or ellipse which includes circular
    truncate : int, optional, default: 4
      truncate the filter at this many standard deviations.

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
    mu = np.zeros(dim)

    # signal
    if shape == "ramp":
        mu = ramp_2D(dim=dim, shape_spec=shape_spec)
    else:
        mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)

    # noise
    noise = np.random.randn(*dim)
    # fwmh = sqrt(8log2)*sigma
    sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

    # smoothing the noise with gaussian kernel
    for i in np.arange(nsubj):
        noise[i, :, :] = gaussian_filter(noise[i, :, :], sigma=sigma_noise, truncate=truncate)
        # rescale the smoothed noise to have sd of the specified sd: std
    noise = noise / np.mean(np.std(noise, 0, ddof=1)) * std
    # print(noise.std(0, ddof=1))
    # print(noise.std(0, ddof=1).mean())

    data = np.array(mu + noise, dtype='float')
    return data, mu

