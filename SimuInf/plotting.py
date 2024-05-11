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
                 cuts=None, label_cut=False, title=None):
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