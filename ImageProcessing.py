from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import glob
import os
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank

# 数据集地址
image_path = r'C:\Users\bunny\Desktop\_Training_Data\sub_mean_images\all/'

# 将所有的图片resize成100*100
w = 2000
h = 2000
c = 3


# 读取图片
def resize_img(path, new_width=0, new_height=0):
    default_height = 500
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    count = 0
    for file_path in file_path_list:
        img = io.imread(file_path)
        _h = img.shape[0]
        _w = img.shape[1]
        new_w = new_width
        new_h = new_height
        if new_w == 0 and new_h == 0:
            new_h = default_height
            new_w = int(_w / _h * new_h)
        if new_w != 0 and new_h == 0:
            new_h = int(_h / _w * new_w)
        if new_w == 0 and new_h != 0:
            new_w = int(_w / _h * new_h)
        img = resize(img, (new_h, new_w), anti_aliasing=True)
        io.imsave(file_path, img)
        del img
        count += 1
        if count % 10 == 0:
            print("\rreading {0}/{1}".format(count, len(file_path_list)), end='')


def stain_separate_image(root_path):
    cate = [root_path + folder for folder in os.listdir(root_path) if os.path.isdir(root_path + folder)]
    count = 0
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            ihc_hed = rgb2hed(img)
            # Rescale hematoxylin and DAB signals and give them a fluorescence look
            _h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1.0))
            _e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 1.0))
            _d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1.0))
            zdh = np.dstack((_h, _e, _d))
            io.imsave(im, zdh)
            del img
            count += 1
            if count % 5000 == 0:
                print(count)


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


def subtract_mean_img(path, disk_size=50):
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    count = 0
    for file_path in file_path_list:
        img = io.imread(file_path) / 255.0
        mean_img = mean_filter(img, disk_size) / 255.0
        sub_mean_img = img - mean_img
        io.imsave(file_path, sub_mean_img)
        del img, mean_img, sub_mean_img
        count += 1
        if count % 10 == 0:
            print("\rreading {0}/{1}".format(count, len(file_path_list)), end='')

# resize_img(image_path, w, h)
# stain_separate_image(image_path)
subtract_mean_img(image_path, 10)