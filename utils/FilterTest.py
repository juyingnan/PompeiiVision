import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.morphology import disk
from skimage.filters import rank

image_path = r'C:\Users\bunny\Desktop\_Training_Data\500_backup\1st\_153_G_Abbate_Casa_di_Sirico_Triclinium_front_part.jpg'
image = io.imread(image_path)/255.0


def mean_filter(img, disk_size=50):
    image_r = img[:, :, 0]
    image_g = img[:, :, 1]
    image_b = img[:, :, 2]
    selem = disk(disk_size)
    # percentile_result = rank.mean_percentile(image, selem=selem, p0=.1, p1=.9)
    # bilateral_result = rank.mean_bilateral(image, selem=selem, s0=500, s1=500)
    normal_result_r = rank.mean(image_r, selem=selem)
    normal_result_g = rank.mean(image_g, selem=selem)
    normal_result_b = rank.mean(image_b, selem=selem)
    normal_result = np.dstack((normal_result_r, normal_result_g, normal_result_b))
    return normal_result


mean_image = mean_filter(image, 10)/255.0

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)
ax = axes.ravel()

titles = ['Original', 'mean', 'normal-mean', 'mean-normal']
imgs = [image, mean_image, image - mean_image, np.absolute(image - mean_image)]
for n in range(0, len(imgs)):
    ax[n].imshow(imgs[n])
    ax[n].set_title(titles[n])
    ax[n].axis('off')

plt.tight_layout()
plt.show()
