import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column  # , column
from bokeh.models import ColumnDataSource, CDSView, IndexFilter, Span, Select, CustomJS  # , CustomJS
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from scipy import io as sio
from sklearn import preprocessing
from skimage import io, transform
import sys
import os


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
            img = transform.rescale(img, 1.0 / 5.0, anti_aliasing=True)
            imgs.append(img)
            labels.append(im.split('\\')[-1])
            if count % 100 == 0:
                print("\rreading {0}/{1}".format(count, min(total_count, len(file_path_list))), end='')
        print('\r', end='')
    return imgs, labels


def generate_feature_list():
    result = []
    append_feature(result, 'Global_color: average_hue_saturation', 3)
    append_feature(result, 'Global_color: hue_distribution', 20)
    append_feature(result, 'Global_color: quantized_hues_number', 1)
    append_feature(result, 'composition: seg_x', 5)
    append_feature(result, 'composition: seg_y', 5)
    append_feature(result, 'composition: seg_variance', 5)
    append_feature(result, 'composition: seg_skewness', 5)
    append_feature(result, 'Segment_color: seg_h', 5)
    append_feature(result, 'Segment_color: seg_s', 5)
    append_feature(result, 'Segment_color: seg_v', 5)
    append_feature(result, 'Key_point: largest n scales of kp', 12)
    return result


def append_feature(result_list, feature_name, feature_count):
    for i in range(feature_count):
        result_list.append(feature_name + ' ' + str(feature_count + 1))


raw_root = r'D:\Projects\pompeii\20190405\svd_500/'
raw_img, raw_file_names = read_img_random(raw_root, 1000)

input_file_name = 'shape_index_10'
x_axis_index = 0
y_axis_index = 1
axis_threshold = 5

if len(sys.argv) >= 4:
    input_file_name = sys.argv[1]
    x_axis_index = int(sys.argv[2])
    y_axis_index = int(sys.argv[3])

mat_path = '../mat/' + input_file_name + '.mat'
digits = sio.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]  # X: nxm: n=67//sample, m=12,10,71,400//feature
file_names, indexes = digits.get('file_name'), digits.get('index')[0]
n_samples, n_features = X.shape
roman_label = ['I', 'II', 'III', 'IV']
markers = ['hex', 'triangle', 'circle', 'cross', 'diamond', 'square', 'x', 'inverted_triangle']
output_file('result/svd_' + input_file_name + '_' + str(x_axis_index) + '_' + str(y_axis_index) + '.html')

# plot tools
tools_list = "pan," \
             "hover," \
             "box_select," \
             "lasso_select," \
             "box_zoom, " \
             "wheel_zoom," \
             "reset," \
             "save," \
             "help"
# highlight axis x & y
vline = Span(location=0, dimension='height', line_color='black', line_width=2)
hline = Span(location=0, dimension='width', line_color='black', line_width=2)

# feature projection calculation
xx_feature_projection_list = list()
xx_feature_correlation_list = list()
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)  # u: mxm, s: mx1, v:nxn/1440x1440
del U
del s
for axis_index in range(axis_threshold):
    ev1 = Vh[axis_index]
    xx_feature_projection_list.append(X.transpose().dot(ev1))

# feature correlation calculation
s_x = preprocessing.normalize(X.transpose())
normalized_vh = preprocessing.normalize(Vh)
for axis_index in range(axis_threshold):
    s_ev1 = normalized_vh[axis_index]
    xx_feature_correlation_list.append(s_x.dot(s_ev1))

# feature data
feature_list = generate_feature_list()
feature_data = {'current_projection_x': xx_feature_projection_list[0],
                'current_projection_y': xx_feature_projection_list[1],
                'current_correlation_x': xx_feature_correlation_list[0],
                'current_correlation_y': xx_feature_correlation_list[1],
                }
# if 'feature' in input_file_name:
#     feature_data[]
feature_source = ColumnDataSource(data=feature_data)

# controls
feature_x = '1'
feature_y = '2'

feature_x_dict = {}
feature_y_dict = {}
for j in range(axis_threshold):
    feature_x_dict[str(j + 1)] = j
    feature_y_dict[str(j + 1)] = j

feature_axis_x_select = Select(value=feature_x, title='X-axis', options=sorted(feature_x_dict.keys()))
feature_axis_y_select = Select(value=feature_y, title='Y-axis', options=sorted(feature_x_dict.keys()))

feature_axis_x_select.js_on_change('value',
                                   CustomJS(args=dict(labels=feature_x_dict, source=feature_source,
                                                      projection_pool=xx_feature_projection_list,
                                                      correlation_pool=xx_feature_correlation_list), code="""
    var index = labels[cb_obj.value];
    console.log(index);
    var data = source.data;
    data['current_projection_x'] = projection_pool[index]
    data['current_correlation_x'] = correlation_pool[index]
    source.change.emit();

    """))
feature_axis_y_select.js_on_change('value',
                                   CustomJS(args=dict(labels=feature_y_dict, source=feature_source,
                                                      projection_pool=xx_feature_projection_list,
                                                      correlation_pool=xx_feature_correlation_list), code="""
    var index = labels[cb_obj.value];
    console.log(index);
    var data = source.data;
    data['current_projection_y'] = projection_pool[index]
    data['current_correlation_y'] = correlation_pool[index]
    source.change.emit();

    """))
feature_controls = column(feature_axis_x_select, feature_axis_y_select)

# feature vis
feature_left = figure(title="features projection", tools=tools_list)
feature_left.xaxis.axis_label = 'Projection on {}'.format(x_axis_index)
feature_left.yaxis.axis_label = 'Projection on {}'.format(y_axis_index)  # highlight x y axes
feature_left.renderers.extend([vline, hline])
feature_left.scatter("current_projection_x", "current_projection_y", source=feature_source, fill_alpha=0.4, size=12)

feature_right = figure(title="features correlation", tools=tools_list)
feature_right.xaxis.axis_label = 'Correlation on {}'.format(x_axis_index)
feature_right.yaxis.axis_label = 'Correlation on {}'.format(y_axis_index)
feature_right.renderers.extend([vline, hline])
feature_right.scatter("current_correlation_x", "current_correlation_y",
                      source=feature_source, fill_alpha=0.4, size=12)

# SAMPLE
xx_sample_projection_list = list()
xx_sample_correlation_list = list()
# sample projection calculation
U, s, Vh = np.linalg.svd(X, full_matrices=False)
for axis_index in range(axis_threshold):
    ev1 = Vh[axis_index]
    xx_sample_projection_list.append(X.dot(ev1))

# sample correlation calculation
s_x = preprocessing.normalize(X)
normalized_vh = preprocessing.normalize(Vh)
for axis_index in range(axis_threshold):
    s_ev1 = normalized_vh[axis_index]
    xx_sample_correlation_list.append(s_x.dot(s_ev1))

# sample data
sample_data = {'current_projection_x': xx_sample_projection_list[0],
               'current_projection_y': xx_sample_projection_list[1],
               'current_correlation_x': xx_sample_correlation_list[0],
               'current_correlation_y': xx_sample_correlation_list[1],
               'style_label': [roman_label[y[i]] for i in range(len(y))],
               'file_name_label': file_names,
               'file_path_label': ["images/" + file_name for file_name in file_names],
               'index_label': indexes,
               }
sample_source = ColumnDataSource(data=sample_data)

# custom_tooltip = [
#     ("style", "@style_label"),
#     ("index", "@index_label"),
#     ("File", "@file_name_label"),
#     # ("(x,y)", "($x, $y)"),
# ]

custom_tooltip = """
    <div>
        <div>
            <img
                src="@file_path_label" height="42" alt="@file_path_label" width="60"
                style="float: left; margin: 0px 15px 15px 0px; image-orientation: from-image;"
            ></img>
        </div>
        <div>
            <span style="font-size: 18px; font-weight: bold;">@style_label</span>
            <span style="font-size: 15px; color: #966;">[@index_label]</span>
        </div>
        <div>
            <span style="font-size: 10px; color: #696;">@file_name_label</span>
        </div>
    </div>
"""

# controls
sample_x = '1'
sample_y = '2'

sample_x_dict = {}
sample_y_dict = {}
for j in range(axis_threshold):
    sample_x_dict[str(j + 1)] = j
    sample_y_dict[str(j + 1)] = j

sample_axis_x_select = Select(value=sample_x, title='X-axis', options=sorted(sample_x_dict.keys()))
sample_axis_y_select = Select(value=sample_y, title='Y-axis', options=sorted(sample_x_dict.keys()))

sample_axis_x_select.js_on_change('value',
                                  CustomJS(args=dict(labels=sample_x_dict, source=sample_source,
                                                     projection_pool=xx_sample_projection_list,
                                                     correlation_pool=xx_sample_correlation_list), code="""
    var index = labels[cb_obj.value];
    console.log(index);
    var data = source.data;
    data['current_projection_x'] = projection_pool[index]
    data['current_correlation_x'] = correlation_pool[index]
    source.change.emit();
    
    """))
sample_axis_y_select.js_on_change('value',
                                  CustomJS(args=dict(labels=sample_y_dict, source=sample_source,
                                                     projection_pool=xx_sample_projection_list,
                                                     correlation_pool=xx_sample_correlation_list), code="""
    var index = labels[cb_obj.value];
    console.log(index);
    var data = source.data;
    data['current_projection_y'] = projection_pool[index]
    data['current_correlation_y'] = correlation_pool[index]
    source.change.emit();

    """))
sample_controls = column(sample_axis_x_select, sample_axis_y_select)


def create_scatter(x_data, y_data, source, title='', x_axis_title='', y_axis_title=''):
    result_plot = figure(title=title, tools=tools_list, tooltips=custom_tooltip)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    for label in roman_label:
        index_list = []
        for i in range(len(source.data['style_label'])):
            if source.data['style_label'][i] == label:
                index_list.append(i)
        view = CDSView(source=source, filters=[IndexFilter(index_list)])
        result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=12,
                            marker=factor_mark('style_label', markers, roman_label),
                            color=factor_cmap('style_label', 'Category10_8', roman_label),
                            # muted_color=factor_cmap(label['real_label_list'], 'Category10_8',
                            #                         label['standard_label_list']),
                            muted_alpha=0.1, view=view,
                            legend=label)
    result_plot.legend.click_policy = "hide"

    # highlight x y axes
    result_plot.renderers.extend([vline, hline])

    return result_plot


# sample vis
sample_plot_list = list()
sample_plot_list.append(
    create_scatter(x_data="current_projection_x", y_data="current_projection_y", source=sample_source,
                   title="samples projection", x_axis_title='Projection on {}'.format(x_axis_index),
                   y_axis_title='Projection on {}'.format(y_axis_index)))
sample_plot_list.append(
    create_scatter(x_data="current_correlation_x", y_data="current_correlation_y", source=sample_source,
                   title="samples correlation", x_axis_title='Correlation on {}'.format(x_axis_index),
                   y_axis_title='Correlation on {}'.format(y_axis_index)))

p = gridplot([[feature_left, feature_right, feature_controls],
              [sample_plot_list[0], sample_plot_list[1], sample_controls]])

show(p)
