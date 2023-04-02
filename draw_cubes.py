import numpy as np
from scipy import io as sio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import os

from wavelength_to_rgb import *


def draw_cubes(img, wavelengths, savedir):
    """
    1. Draw and save image of each channel of Hyperspectral image with color map according to wavelength.
    2. Draw and save all channels of Hyperspectral image in one image, with wavelength text in left-top of each sub-image

    img: Hyperspectral image which shape should be (H, W, C). e.g (512, 512, 31).
    wavelengths: List of wavelength of each channel of Hyperspectral images.
    savedir: The path to store images. 
    """
    if img.shape[-1] != len(wavelengths):
        raise ValueError("Channels number must equal to length of array 'wavelengths'")
    
    for i in range(len(wavelengths)):
        color = wavelength_to_rgb(wavelengths[i])
        colors = [(0, 0, 0), color]
        cmap = LinearSegmentedColormap.from_list('my_colormap', colors)
        fig = plt.figure()
        
        img_temp = img[:, :, i]
        vmin = np.min(img_temp)
        vmax = np.max(img_temp)
        img_temp = (img_temp - vmin) / (vmax - vmin)
        
        plt.imshow(img_temp, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        
        if os.path.exists(savedir) is False:
            os.makedirs(savedir)

        savepath = savedir + "/channel-" + str(i) + ".png"
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    line_num = int(np.ceil(len(wavelengths) / 7))
    fig = plt.figure(figsize=(30, line_num * 4))

    for i in range(len(wavelengths)):
        color = wavelength_to_rgb(wavelengths[i])
        colors = [(0, 0, 0), color]
        cmap = LinearSegmentedColormap.from_list('my_colormap', colors)

        plt.subplot(line_num, 7, i+1)
        img_temp = img[:, :, i]
        vmin = np.min(img_temp)
        vmax = np.max(img_temp)
        img_temp = (img_temp - vmin) / (vmax - vmin)
        
        plt.imshow(img_temp, cmap=cmap, vmin=0, vmax=1)
        plt.text(30, 80, str(round(wavelengths[i], 1)) + " nm", fontsize=14, color="white", fontweight="bold")
        plt.axis('off')

    plt.tight_layout()
    savepath = savedir + "/all-channels" + ".png"
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    imgpath = "result_img/20230331/fxny_ADMM-HSICNN_result_20230331193936.mat"
    savedir = "test"

    img = sio.loadmat(imgpath)['img']

    lam28 = [674.5968, 660.1400, 646.2697, 632.9423, 620.2400, 608.0095, 596.2223, 585.1468, 574.9737, 565.3837, 556.2646, 547.5692, 539.2501, 531.2600, 523.5644, 516.1539, 509.0188, 502.1495, 495.5365, 489.1700, 483.0124, 477.0366, 471.2485, 465.6544, 460.2602, 455.0721, 450.0971, 445.3559]

    draw_cubes(img, lam28, "test")


if __name__ == '__main__':
    main()
