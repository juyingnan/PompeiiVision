import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from PIL import Image
from skimage import morphology
from skimage.morphology import disk
import csv
from skimage import color


def removeSmallObject(img, minSize=500):
    cleanedImg = morphology.remove_small_holes(img, min_size=minSize)
    return np.asarray(cleanedImg, np.int8) * 255


def morphology_test(img):
    selem = disk(1)
    cleanedImg = morphology.closing(img, selem)
    return cleanedImg


def wharershed_segmentation_floor(image):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False,
                                footprint=np.ones((50, 50)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image, watershed_line=True)
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax[2].set_title('Separated objects')
    from skimage.measure import regionprops
    import matplotlib.patches as mpatches
    result = []
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            result.append(list(region.bbox))
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax[2].add_patch(rect)
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()
    plt.show()
    # add size and reverse sort by size
    for i in range(len(result)):
        minr, minc, maxr, maxc = result[i]
        size = (maxc - minc) * (maxr - minr)
        result[i].append(size)
    return sorted(result, key=lambda l: l[4], reverse=True)


def wharershed_segmentation_wall(image):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False,
                                # footprint=np.ones((50, 50)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image, watershed_line=True)
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax[2].set_title('Separated objects')
    from skimage.measure import regionprops
    import matplotlib.patches as mpatches
    result = []
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 10:
            # draw rectangle around segmented coins
            result.append(region.bbox)
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax[2].add_patch(rect)
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()
    plt.show()
    return result


def write_csv(path, lines, header=None):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            writer.writerow((header))
        writer.writerows(lines)


# Constructing test image
# path = r'C:\Users\bunny\Desktop\pompeii\test\test8.jpg'
path = r'C:\Users\bunny\Desktop\pompeii\20180821\6 08 05 plan.jpg'
img = Image.open(path)  # .transpose(Image.FLIP_TOP_BOTTOM)
# img = img.resize((1160, 1636), Image.NEAREST)
gray = np.asarray(img.convert('L'))
threshold = 96
thresholdedData = (gray > threshold) * 255
thresholdedData = morphology_test(thresholdedData)
thresholdedData = removeSmallObject(thresholdedData, minSize=100)
shresholdedData_invert = 255 - thresholdedData
# from scipy import ndimage
# image = ndimage.interpolation.rotate(image, 25)
floors = wharershed_segmentation_floor(thresholdedData)
walls = wharershed_segmentation_wall(shresholdedData_invert)
# save to csv
floor_header = ["minr", "minc", "maxr", "maxc", "size"]
wall_header = ["minr", "minc", "maxr", "maxc"]
csv_floor_path = path.split('.')[0] + '_floor.csv'
write_csv(csv_floor_path, floors, floor_header)
csv_wall_path = path.split('.')[0] + '_wall.csv'
write_csv(csv_wall_path, walls, wall_header)
