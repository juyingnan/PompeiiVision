import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

image_left = io.imread(r'C:\Users\bunny\Desktop\june.png')
image_right = io.imread(r'C:\Users\bunny\Desktop\december.png')

# extract corners using Harris' corner measure
coords_left = corner_peaks(corner_harris(image_left), threshold_rel=0.5, min_distance=64)
coords_right = corner_peaks(corner_harris(image_right), threshold_rel=0.5, min_distance=64)

# determine sub-pixel corner position
coords_left_subpix = corner_subpix(image_left, coords_left, window_size=9)
coords_right_subpix = corner_subpix(image_right, coords_right, window_size=9)


def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext + 1, -window_ext:window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma ** 2))
    g /= 2 * np.pi * sigma * sigma
    return g


def match_corner(coord, window_ext=5):
    r, c = np.round(coord).astype(np.intp)
    window_left = image_left[r - window_ext:r + window_ext + 1, c - window_ext:c + window_ext + 1]

    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    # weights = np.dstack((weights, weights))

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_right_subpix:
        if cr > 0 and cc > 0:
            cr = np.int(cr)
            cc = np.int(cc)
            window_right = image_right[cr - window_ext:cr + window_ext + 1, cc - window_ext:cc + window_ext + 1]
            SSD = np.sum(weights * (window_left - window_right) ** 2)
            SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return coords_right_subpix[min_idx]


# find correspondences using simple weighted sum of squared differences
src = []
dst = []
for coord in coords_left_subpix:
    if coord[0] > 0 and coord[1] > 0:
        src.append(coord)
        dst.append(match_corner(coord))
src = np.array(src)
dst = np.array(dst)

# estimate affine transform model using all coordinates
model = AffineTransform()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
outliers = inliers == False

# visualize correspondence
fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

inlier_idxs = np.nonzero(inliers)[0]
plot_matches(ax[0], image_left, image_right, src, dst, np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
ax[0].axis('off')
ax[0].set_title('Correct correspondences')

outlier_idxs = np.nonzero(outliers)[0]
plot_matches(ax[1], image_left, image_right, src, dst,
             np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
ax[1].axis('off')
ax[1].set_title('Faulty correspondences')

plt.show()
