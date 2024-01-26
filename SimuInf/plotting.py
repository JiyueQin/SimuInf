import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from nilearn.image import get_data

def confset_plot(confset_ls, name_ls, nrow=1, ncol=None, fontsize=20,
                 figsize=(30, 20), ticks=True, background=None,
                 cut=None, label_cut=False, title=None):
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
        confset_ls = [[np.flipud(np.transpose(data[:, :, cut])) for data in list(confset) + [get_data(background)]] for
                      confset in confset_ls]
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
    # print(axs.shape)

    # each subplot
    for i in range(nrow):
        for j in range(ncol):
            if k == n:
                break
            if not ticks:
                axs[i, j].get_xaxis().set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)

            if background is not None:
                axs[i, j].imshow(confset_ls[k][3], alpha=1, cmap="gray")

            # black/blue for the outer set
            if np.sum(confset_ls[k][2] == 0) == 0:
                axs[i, j].imshow(confset_ls[k][2], cmap=colors.ListedColormap(['blue', 'black']))
            else:
                axs[i, j].imshow(confset_ls[k][2], cmap=cmap1)

            # none/yellow for the estimated set
            axs[i, j].imshow(confset_ls[k][0], cmap=cmap2)

            # none/red for the inner set
            axs[i, j].imshow(confset_ls[k][1], cmap=cmap3)

            # title of the subplot
            axs[i, j].set_title(name_ls[k], fontsize=fontsize)
            k = k + 1

    plt.suptitle(title)
    # plt.show()


def ls_plot(image_ls, name_ls=None, nrow=1, ncol=None, fontsize=20, figsize=(30, 20),
            title=None, titlesize=20,
            colorbar=None, ticks=True):
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
    # each subplot
    for i in range(nrow):
        for j in range(ncol):
            if k == n:
                break
            im = axs[i, j].imshow(image_ls[k])
            if not ticks:
                axs[i, j].get_xaxis().set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)

            # title of the subplot
            if name_ls is not None:
                axs[i, j].set_title(name_ls[k], fontsize=fontsize)
            k = k + 1
            if colorbar == 'individual':
                plt.colorbar(im)
    if colorbar == 'share':
        # [left, bottom, width, height] of the new axes
        cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.5])
        fig.colorbar(im, cax=cbar_ax)
    plt.suptitle(title, size=titlesize)

    # plt.show()

