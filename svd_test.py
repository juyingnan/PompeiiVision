# 1. read mat & restore the first image
# 2. get COV matrix
# 3. get eigenvectors
# 4. rebuild the matrix
# 5. scatter the eigenvectors

import matplotlib.pyplot as plt
from scipy import io as sio
import numpy as np
from sklearn import preprocessing
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import os
from skimage import io, transform
import xml.etree.ElementTree as ET
from io import BytesIO


def getImage2(img, zoom=1.0):
    return OffsetImage(img, zoom=zoom)


def read_img_random(path, total_count):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % folder)
        count = 0
        file_path_list = [os.path.join(folder, file_name) for file_name in os.listdir(folder)
                          if os.path.isfile(os.path.join(folder, file_name))]
        while count < total_count and count < len(file_path_list):
            im = file_path_list[count]
            count += 1
            img = io.imread(im)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = transform.rescale(img, 1.0 / 4.0, anti_aliasing=True)
            imgs.append(img)
            labels.append(im.split('\\')[-1])
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return imgs, labels


raw_root = r'D:\Projects\pompeii\test_20180919\svd_test/'
raw_img, raw_file_names = read_img_random(raw_root, 1000)
ET.register_namespace("", "http://www.w3.org/2000/svg")

mat_path = 'mat/raw_20.mat'
x_axis_index = 0
y_axis_index = 1
digits = sio.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]  # X: nxm: n=67//sample, m=12,10,71,400//feature
file_names, indexes = digits.get('file_name'), digits.get('index')[0]
n_samples, n_features = X.shape
roman_label = ['I', 'II', 'III', 'IV']

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))  # values: nx1/67x1, vectors: nxn/67x67

U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=True)  # u: mxm, s: mx1, v:nxn/67x67
# s[2:] = 0

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(hspace=0.35)
ax = fig.add_subplot(321)
ax.imshow(X.transpose())
if "raw" in mat_path:
    ax.set_aspect(0.1)
ax.set_title('original_mat')

ax = fig.add_subplot(322)
ax.bar(np.arange(len(s)), s)
ax.set_title('singular_values_feature')

small_edge_index = 0.2
ax = fig.add_subplot(323)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Projection on {}'.format(x_axis_index + 1))
plt.ylabel('Projection on {}'.format(y_axis_index + 1))
ev1 = Vh.transpose()[x_axis_index].real  # ev: nx1/67x1
ev2 = Vh.transpose()[y_axis_index].real
xx = X.transpose().dot(ev1)  # mxn.nx1 = mx1
yy = X.transpose().dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    if "raw" in mat_path:
        ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
    else:
        ax.text(xx[i], yy[i], str(i), fontdict={'size': 8})
ax.set_title('features_projection')

ax = fig.add_subplot(324)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Correlation on {}'.format(x_axis_index + 1))
plt.ylabel('Correlation on {}'.format(y_axis_index + 1))
s_x = preprocessing.normalize(X.transpose())
normalized_vh = preprocessing.normalize(Vh.real.transpose())
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx = s_x.dot(s_ev1)
yy = s_x.dot(s_ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    if "raw" in mat_path:
        ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
    else:
        ax.text(xx[i], yy[i], str(i), fontdict={'size': 8})
ax.set_title('features_correlation')

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))  # values: mx1/12x1, vectors: mxm/12x12
U, s, Vh = np.linalg.svd(X, full_matrices=True)  # u: nxn/67x67, s: mx1, v:mxm
# s[2:] = 0

lo = 0.1  # label_offset

ax = fig.add_subplot(325)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Projection on {}'.format(x_axis_index + 1))
plt.ylabel('Projection on {}'.format(y_axis_index + 1))
ev1 = Vh.transpose()[x_axis_index].real  # ev: nx1/67x1
ev2 = Vh.transpose()[y_axis_index].real
xx = X.dot(ev1)  # nxm.mx1=nx1
yy = X.dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)

# for i in range(5):
for i in range(X.shape[0]):
    # dot and text
    ax.text(xx[i] - 3 * lo, yy[i] + lo, roman_label[y[i]] + '-' + str(indexes[i]), color=plt.cm.tab10(int(y[i])),
            fontdict={'size': 8})
    dot = plt.Circle((xx[i], yy[i]), 0.05, color=plt.cm.tab10(int(y[i])))
    patch = ax.add_patch(dot)
    ax.add_patch(patch)
    patch.set_gid('mypatch_{:03d}'.format(i))

    # image
    label = str(file_names[i]).strip()
    index = raw_file_names.index(label)
    imagebox = getImage2(raw_img[index])
    ab = AnnotationBbox(imagebox, xy=(xx[i] + lo, yy[i] + lo), frameon=False, box_alignment=(0, 0))
    ab.offsetbox.get_children()[0].set_gid('myimage_{:03d}'.format(i))
    ax.add_artist(ab)

    # tooltip
    annotate = ax.annotate(label, xy=(xx[i], yy[i] - 3 * lo), xytext=(0, 0),
                           textcoords='offset points', color='w', ha='center',
                           fontsize=9, bbox=dict(boxstyle='round, pad=.5',
                                                 fc=(.1, .1, .1, .92),
                                                 ec=(1., 1., 1.), lw=1,
                                                 zorder=3))
    annotate.set_gid('mytooltip_{:03d}'.format(i))

ax.set_title('samples/images_projection')

ax = fig.add_subplot(326)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Correlation on {}'.format(x_axis_index + 1))
plt.ylabel('Correlation on {}'.format(y_axis_index + 1))
s_x = preprocessing.normalize(X)
normalized_vh = preprocessing.normalize(Vh.real.transpose())
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx = s_x.dot(s_ev1)
yy = s_x.dot(s_ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(X.shape[0]):
    ax.text(xx[i], yy[i], roman_label[y[i]] + '-' + str(indexes[i]), color=plt.cm.tab10(int(y[i])),
            fontdict={'size': 8})
    print(roman_label[y[i]] + '-' + str(indexes[i]), file_names[i])
ax.set_title('samples/images_correlation')

# plt.show()

f = BytesIO()
fig.savefig(f, format="svg")

# --- Add interactivity ---

# Create XML tree from the SVG file.
tree, xmlid = ET.XMLID(f.getvalue())
tree.set('onload', 'init(evt)')

# print(xmlid)

# for i in range(5):
for i in range(X.shape[0]):
    # Hide the tooltips
    tooltip = xmlid['mytooltip_{:03d}'.format(i)]
    tooltip.set('visibility', 'hidden')
    imagetip = xmlid['myimage_{:03d}'.format(i)]
    imagetip.set('visibility', 'hidden')
    # Assign onmouseover and onmouseout callbacks to patches.
    mypatch = xmlid['mypatch_{:03d}'.format(i)]
    mypatch.set('onmouseover', "ShowTooltip(this)")
    mypatch.set('onmouseout', "HideTooltip(this)")

# This is the script defining the ShowTooltip and HideTooltip functions.
script = """
    <script type="text/ecmascript">
    <![CDATA[

    function init(evt) {
        if ( window.svgDocument == null ) {
            svgDocument = evt.target.ownerDocument;
            }
        }

    function ShowTooltip(obj) {
        var cur = obj.id.split("_")[1];
        var tip = svgDocument.getElementById('mytooltip_' + cur);
        tip.setAttribute('visibility',"visible");
        var img = svgDocument.getElementById('myimage_' + cur);
        img.setAttribute('visibility',"visible")
        }

    function HideTooltip(obj) {
        var cur = obj.id.split("_")[1];
        var tip = svgDocument.getElementById('mytooltip_' + cur);
        tip.setAttribute('visibility',"hidden");
        var img = svgDocument.getElementById('myimage_' + cur);
        img.setAttribute('visibility',"hidden")
        }

    ]]>
    </script>
    """

# Insert the script at the top of the file and save it.
tree.insert(0, ET.XML(script))
ET.ElementTree(tree).write('svg_tooltip.svg')
