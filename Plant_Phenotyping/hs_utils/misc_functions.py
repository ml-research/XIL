import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
from sklearn import preprocessing
from scipy import ndimage as ndi
from skimage import feature

mpl.rcParams['savefig.pad_inches'] = 0


def apply_colormap_on_image(org_im, activation, colormap_name, alpha=.4):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = alpha#0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_class_activation_images3D(org_img, activation_map, file_name, samples, spectral_activations, orig_wavelength, wv_plot=True):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    # Grayscale activation map

    if wv_plot:
        fig, axs = plt.subplots(ncols=activation_map.shape[0]+1, nrows=2, figsize=(12, 20))
        axs = axs.flatten()
        gs = axs[0].get_gridspec()
        for ax in axs[0:activation_map.shape[0]+1]:
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.remove()

        axbig = fig.add_subplot(gs[1:activation_map.shape[0]+1])
        axbig.set_xticklabels([])
        fig.tight_layout()

        wavelength_area = len(samples[0]) // activation_map.shape[0]
        axbig.axvline(x=wavelength_area, lw=2, ls="--", c="#000000", alpha=0.4)
        axbig.axvline(x=wavelength_area * 2, lw=2, ls="--", c="#000000", alpha=0.4)
        axbig.axvline(x=wavelength_area * 3, lw=2, ls="--", c="#000000", alpha=0.4)
        axbig.tick_params(axis='both', which='major', labelsize=20)
        axbig.axes.get_xaxis().set_visible(False)
        axs1_twin = axbig.twinx()
        axs1_twin.tick_params(axis='both', which='major', labelsize=20)
        axs1_twin.plot([wavelength_area - wavelength_area / 2,
                        wavelength_area * 2 - wavelength_area / 2,
                        wavelength_area * 3 - wavelength_area / 2,
                        wavelength_area * 4 - wavelength_area / 2],
                       spectral_activations[0],
                       lw=8,
                       c="#6A1B9A",
                       alpha=0.6)
        axs1_twin.errorbar([wavelength_area - wavelength_area / 2,
                            wavelength_area * 2 - wavelength_area / 2,
                            wavelength_area * 3 - wavelength_area / 2,
                            wavelength_area * 4 - wavelength_area / 2],
                           spectral_activations[0], spectral_activations[1], c="#BA68C8", linestyle='None',
                           ms=20, fmt='o', capsize=40, alpha=0.6)
        for sample in samples[:2]:
            axbig.plot(sample, lw=4, ls="--", alpha=0.6)
            #xticks = orig_wavelength[::3][::10]
        axs1_twin.set_ylim((0, 1))
        axbig.set_ylim((0, 1))
        axbig.set_xlim((0, len(samples[0])))
        axbig.yaxis.set_major_locator(MaxNLocator(prune='both'))
        axs1_twin.yaxis.set_major_locator(MaxNLocator(prune='both'))
        axs_next = axs[-activation_map.shape[0]-1:]
    else:
        fig, axs = plt.subplots(ncols=activation_map.shape[0]+1, nrows=1, figsize=(12, 20))
        axs_next = axs
    #axbig.set_title('Wavelength Activation Map', fontsize=30)
    # axs[1].axis('off')

    for idx, ax in enumerate(axs_next):
        #heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map[idx], 'viridis', alpha=1.0)
        #print(np.min(activation_map), np.max(activation_map))
        if idx != 0:
            org_img_edged = preprocessing.scale(np.array(org_img, dtype=float)[:,:,1]/255)
            org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
            # Compute the Canny filter for two values of sigma
            org_img_edged = feature.canny(org_img_edged, sigma=3)
            ax.imshow(org_img_edged, cmap=plt.cm.binary)
            ax.imshow(activation_map[idx-1], cmap='viridis', vmin=np.min(activation_map), vmax=np.max(activation_map),
                      alpha=0.4)
        else:
            ax.imshow(np.array(org_img))
        #ax.set_title('CAM Heatmap {}'.format(idx), fontsize=30)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    #plt.show()
    plt.subplots_adjust(top=0.25, wspace=0, hspace=0)
    fig.savefig(file_name, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    org_img_edged = preprocessing.scale(np.array(org_img, dtype=float)[:, :, 1] / 255)
    org_img_edged = ndi.gaussian_filter(org_img_edged, 4)
    # Compute the Canny filter for two values of sigma
    org_img_edged = feature.canny(org_img_edged, sigma=3)
    plt.imshow(org_img_edged, cmap=plt.cm.binary)
    plt.imshow(np.max(activation_map, axis=(0,)), cmap='viridis', alpha=0.4)
    plt.savefig(file_name.replace(".jpg", "_mean.jpg"), bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()


def save_class_activation_images2D(org_img, activation_map, dir_name, file_name, true_class, pred_class):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'viridis')

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    ax[0, 0].imshow(org_img)
    ax[0, 0].set_title('Original RGB', fontsize=30)
    ax[0, 0].axis('off')

    ax[0, 1].imshow(activation_map, cmap='gray')
    ax[0, 1].set_title('CAM Grayscale', fontsize=30)
    ax[0, 1].axis('off')

    im10 = ax[1, 0].imshow(heatmap)
    ax[1, 0].set_title('CAM Heatmap', fontsize=30)
    ax[1, 0].axis('off')

    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im10, cax=cax, orientation='vertical')

    ax[1, 1].imshow(heatmap_on_image)
    ax[1, 1].set_title('CAM Heatmap overlay', fontsize=30)
    ax[1, 1].axis('off')

    plt.suptitle('True Class: ' + str(true_class) + '; Pred Class: ' + str(pred_class) + '; CAM Class: ' +
                 file_name.split('_')[-1], fontsize=30)

    path_to_file = os.path.join('../results' + dir_name, file_name + '_y_' + str(true_class) + '_ypred_' +
                                str(pred_class) + '_Cam_Heatmap.png')
    fig.savefig(path_to_file, bbox_inches='tight')
    plt.close()
