import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage import io
from skimage.feature import canny

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import measure
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse


def find_contour(img):
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.8)
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def classic_hough_line(img):
    # Classic straight-line Hough transform
    h, theta, d = hough_line(img)
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    ax[2].imshow(img, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, img.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')
    plt.tight_layout()
    plt.show()


def prob_hough_line(img):
    #edges = canny(img, sigma=4)
    edges = canny(img, sigma=5, low_threshold=0.05, high_threshold=0.10)
    lines = probabilistic_hough_line(edges, threshold=50, line_length=20,
                                     line_gap=10)
    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')
    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')
    for a in ax:
        a.set_axis_off()
    plt.tight_layout()
    plt.show()

def canny_test(img):
    edges = canny(img, sigma=5, low_threshold=0.05, high_threshold=0.10)

    # Generating figure 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')
    for a in ax:
        a.set_axis_off()
    plt.tight_layout()
    plt.show()

def detect_corner(img):

    coords = corner_peaks(corner_harris(img), min_distance=5)
    coords_subpix = corner_subpix(img, coords, window_size=13)

    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    #ax.axis((0, 350, 350, 0))
    plt.show()

# Constructing test image
path = r'C:\Users\bunny\Desktop\test3.png'
image = io.imread(path, as_grey=True)

#classic_hough_line(image)
#find_contour(image)
#canny_test(image)
prob_hough_line(image)
#detect_corner(image)
