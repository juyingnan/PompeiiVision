from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import glob
import os
import numpy as np

# 数据集地址
image_path = r'C:\Users\bunny\Desktop\km\2000/'

# 将所有的图片resize成100*100
w = 2000
h = 2000
c = 3


# 读取图片
def resize_img(path, new_width, new_height):
    file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)
                      if os.path.isfile(os.path.join(path, file_name))]
    count = 0
    for file_path in file_path_list:
        img = io.imread(file_path)
        img = resize(img, (new_width, new_height))
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


resize_img(image_path, w, h)
# stain_separate_image(image_path)
