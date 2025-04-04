def return_cuts(confset_ls, display_mode, cut, background):
    if display_mode == 'z':
        # get the brain image into the right direction
        confset_ls = [[np.flipud(np.transpose(data[:, :, cut])) for data in list(confset) + [get_data(background)]] for
                      confset in confset_ls]
    if display_mode == 'x':
        confset_ls = [[np.flipud(np.transpose(data[cut, :, :])) for data in list(confset) + [get_data(background)]] for
                      confset in confset_ls] 
    if display_mode == 'y':
        confset_ls = [[np.flipud(np.transpose(data[:, cut, :])) for data in list(confset) + [get_data(background)]] for
                      confset in confset_ls]        
    return confset_ls


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from nilearn.image import get_data

def confset_plot(confset_ls, name_ls, nrow=1, ncol=None, fontsize=20,
                 figsize=(30, 20), ticks=True, background=None, display_mode='z', 
                 cuts=None, label_cut=False, title=None, truth_mask_ls = None,
                 contour_color = 'purple', contour_linewidth = 1.5, alpha = 1):
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
    display_mode: string, optional, default: 'z'
      the direction to slice the 3d image, options are 'x', 'y', 'z'
    cuts:list or an integer, optional, default: None
      a list of coordinates to slice for a 3d image, or an integer if there is only one cut 
    label_cut: bool, optional, default: False
      if True, the title will include the z coordinate for the slice

    Examples
    --------
    confset_plot(confset_ls, name_ls)

    """
    n = len(confset_ls)
    #print(n)
    if cuts is not None: 
        if isinstance(cuts, int):
            cuts = [cuts]
        if label_cut:
            cuts_labels_ls = list(np.repeat(cuts, n))
        if len(cuts)>1:
            n = n*len(cuts)
            name_ls = name_ls*len(cuts)
    k = 0
    if ncol is None:
        ncol = np.ceil(n / nrow).astype(int)
    # to do: here assumes background means 3D, update 
    if background is None:
        cmap1 = colors.ListedColormap(['black', 'blue'])
    else:
        cmap1 = colors.ListedColormap(['none', 'blue'])
        confset_2d_ls = []
        for cut in cuts:
            confset_2d = return_cuts(confset_ls, display_mode=display_mode, cut=cut, background=background)
            #print(len(confset_2d))
            confset_2d_ls = confset_2d_ls + confset_2d
        confset_ls = confset_2d_ls
        #print(len(confset_ls))
        if label_cut:
            name_ls = [name_cut[0]+f", {display_mode}={name_cut[1]}" for name_cut in zip(name_ls, cuts_labels_ls)]
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
                axs[i, j].imshow(confset_ls[k][2], cmap=colors.ListedColormap(['blue', 'black']), alpha = alpha)
            else:
                axs[i, j].imshow(confset_ls[k][2], cmap=cmap1, alpha = alpha)

            # none/yellow for the estimated set
            axs[i, j].imshow(confset_ls[k][0], cmap=cmap2, alpha = alpha)

            # none/red for the inner set
            axs[i, j].imshow(confset_ls[k][1], cmap=cmap3, alpha = alpha)

            if truth_mask_ls is not None:
                # Draw contour around the mask
                axs[i, j].contour(truth_mask_ls[k], levels=[0.5], colors=contour_color, 
                                  linestyles='dashed', linewidths=contour_linewidth)

            # title of the subplot
            axs[i, j].set_title(name_ls[k], fontsize=fontsize)
            k = k + 1

    plt.suptitle(title)
    # plt.show()




def ls_plot(image_ls, name_ls=None, nrow=1, ncol=None, fontsize=20, figsize=(30, 20),
            title=None, titlesize=20,
            colorbar=None, ticks=True, colorbar_location = 'right'):
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
        if colorbar_location == 'right':
        # [left, bottom, width, height] of the new axes [left, bottom]:coordinate of the southwest corner
        # All quantities are in fractions of figure width and height.
        # note, there can be white regions in the figure
            cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.5])
            fig.colorbar(im, cax=cbar_ax)
        elif colorbar_location == 'left':
            cbar_ax = fig.add_axes([0.15, 0.2, 0.015, 0.5])
            fig.colorbar(im, cax=cbar_ax)
        else:
            fig.colorbar(im, location = 'top')
        if colorbar_location == 'left':
            cbar_ax.yaxis.set_ticks_position('left')

    plt.suptitle(title, size=titlesize)

    # plt.show()

