import numpy as np
from scipy.ndimage import gaussian_filter
from confidenceset.random_field_generator import ramp_2D, ellipse_2D


def gen_2D(dim, shape, shape_spec, truncate=4, noise_type='gaussian', noise_df=3):
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
        noise_raw = np.random.standard_t(noise_df, dim_larger)
        if noise_df > 2:
            sd_before_smoothing = np.sqrt(noise_df / (noise_df - 2))
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

    # sd across all subjects for each pixel
    # print(noise.std(0, ddof=1))
    # print(noise.std(0, ddof=1).mean())
    # print(noise.std(0, ddof=1).std())
    data = np.array(mu + noise, dtype='float')
    return data, mu