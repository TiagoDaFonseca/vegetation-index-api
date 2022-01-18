import cv2
import cmapy
import numpy as np
import logging
from colormap import Colormap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


logger = logging.getLogger("Processing module")
logging.basicConfig(level=logging.DEBUG)

THEMES = ['rainbow',
          'brg',
          'afmhot',
          'Spectral',
          'Greys',
          'Greens',
          'Oranges',
          'Reds',
          'Blues'
          'spring',
          'summer',
          'autumn',
          'winter',
          'hot',
          'spotlight',
          'spotlight_r',
          'rainbow_r',
          'brg_r',
          'afmhot_r',
          'Spectral_r',
          'Greys_r',
          'Greens_r',
          'Oranges_r',
          'Reds_r',
          'Blues_r'
          'spring_r',
          'summer_r',
          'autumn_r',
          'winter_r',
          'hot_r',
          ]


def apply_colormap(image: np.array, theme=None) -> np.array:
    assert theme is not None, logger.debug("Please choose a theme.")

    out = image.astype(np.uint8)
    if theme == 'spotlight':
        cmap = cmapy.cmap('RdYlGn')
    elif theme == 'spotlight_r':
        cmap = cmapy.cmap('RdYlGn_r')
    else:
        cmap = cmapy.cmap(theme)

    return cv2.applyColorMap(out, cmap)


def apply_heatmap(data: np.array, min_val=None, max_val=None, center_val=None, cm=None, vi=None):
    assert min_val is not None, logger.debug("Please set min value")
    assert max_val is not None, logger.debug("Please set max value")

    if cm == 'spotlight':
        cmap = Colormap().cmap('RdYlGn', reverse=False)
    elif cm == 'spotlight_r':
        cmap = Colormap().cmap('RdYlGn', reverse=True)
    else:
        if "_r" in cm:
            c = cm.split('_')[0]
            cmap = Colormap().cmap(c, reverse=True)
        else:
            cmap = Colormap().cmap(cm, reverse=False)

    if center_val is None:
        center_val = int((min_val + max_val)/2)
    fig, ax = plt.subplots()
    sns.heatmap(data=data, cmap=cmap, vmin=min_val, vmax=max_val, center=center_val, ax=ax)
    plt.xticks([])
    plt.yticks([])
    if vi is not None:
        plt.xlabel(vi)
    return fig


def plot_severity(data: np.array):
    cmap = ListedColormap(['black', 'green', 'gold', 'darkorange', 'red'])
    fig, ax = plt.subplots()
    sns.heatmap(data=data.reshape(data.shape[0], data.shape[1]), cmap=cmap, vmin=-0.5, vmax=4.5, ax=ax, cbar=False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Vegetation Vigour", color="gray", size=15)
    return fig


# -- Unit test -- #
if __name__ == "__main__":
    normal_data = np.random.randn(16,18)
    fig = apply_heatmap(normal_data, cm='brg_r', min_val=-10, max_val=10, center_val=0, vi="boda")
    plt.show()
