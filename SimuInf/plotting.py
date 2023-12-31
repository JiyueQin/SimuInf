import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from nilearn.image import get_data

def confset_plot(confset_ls, name_ls, nrow=1, ncol=None, fontsize=20, figsize=(30, 20), background=None, cut=None, label_cut=False):
    """
    plot a list of confidence sets

    Parameters
    ----------
    confset_ls: list
      a list of tuples for confidence sets(est_set, inner_set, outer_set)
    name_ls: list
      a list of names for each subplot
    nrow: int, optional, default: 1
      number of rows of the figure
    ncol: int, optional, default: the length of confset_ls/nrow
    fontsize : int, optional, default: 20
      font size for figure
    figsize : tuple, optional, default: (30,20)
      figure size
    background: nifti image, optional, default: None
      the background image
    cut:int, optional, default: None
      the z coordinate to slice for a 3d image
    label_cut: bool, optional, default: False
      if True, the title will include the z coordinate for the slice

    Examples
    --------
    confset_plot(confset_ls, name_ls)

    """

    n = len(confset_ls)
    k = 0
    if ncol is None:
        ncol = np.ceil(n / nrow).astype(int)
    if background is None:
        cmap1 = colors.ListedColormap(['black', 'blue'])
    else:
        cmap1 = colors.ListedColormap(['none', 'blue'])
        # get the brain image into the right direction
        confset_ls = [[np.flipud(np.transpose(data[:,:,cut])) for data in list(confset)+[get_data(background)]] for confset in confset_ls]
        if label_cut:
            name_ls = [name + f",cut at z={cut}" for name in name_ls]
    cmap2 = colors.ListedColormap(['none', 'yellow'])
    cmap3 = colors.ListedColormap(['none', 'red'])

    # a plot with multiple subplots
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    # print(axs)
    # print(type(axs))
    if nrow == 1:
        axs = np.array([axs]).reshape((1, -1))
    elif ncol == 1:
        axs = np.array([axs]).reshape((-1, 1))
    #print(axs.shape)

    ## each subplot
    for i in range(nrow):
        for j in range(ncol):
            if k == n:
                break
            if background is not None:
                axs[i, j].imshow(confset_ls[k][3], alpha=1, cmap="gray")

            ## black/blue for the outer set
            axs[i, j].imshow(confset_ls[k][2], cmap=cmap1)

            ## none/yellow for the estimated set
            axs[i, j].imshow(confset_ls[k][0], cmap=cmap2)

            ## none/red for the inner set
            axs[i, j].imshow(confset_ls[k][1], cmap=cmap3)

            ## title of the subplot
            axs[i, j].set_title(name_ls[k], fontsize=fontsize)
            k = k + 1

    #plt.suptitle(f"method={method}, confset method={temp}, alpha={alpha}")
    #plt.show()


def ls_plot(image_ls, name_ls, nrow=1, ncol=None, fontsize=20, figsize=(30, 20), title=None, titlesize=20):
    """
    plot a list of 2D images

    Parameters
    ----------
    image_ls: list
        a list of 2D arrays
    name_ls: list
        a list of names for each subplot
    nrow: int, optional, default: 1
        number of rows of the figure
    ncol: int, optional, default: the length of signal_ls/nrow
    fontsize : int, optional, default: 20
        font size for figure
    figsize : tuple, optional, default: (30,20)
        figure size

    Examples
    --------
    ls_plot(signal_ls, signal_name_ls)

    """
    n = len(image_ls)
    # print(n)
    k = 0
    if ncol is None:
        ncol = np.ceil(n / nrow).astype(int)
        # print(ncol)
    # a plot with multiple subplots
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    # print(axs.shape)
    if nrow == 1:
        axs = np.array([axs]).reshape((1, -1))
    elif ncol == 1:
        axs = np.array([axs]).reshape((-1, 1))
    ## each subplot
    for i in range(nrow):
        for j in range(ncol):
            if k == n:
                break
            axs[i, j].imshow(image_ls[k])

            ## title of the subplot
            axs[i, j].set_title(name_ls[k], fontsize=fontsize)
            k = k + 1

    plt.suptitle(title, size=titlesize)
    plt.show()
