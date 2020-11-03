from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CDSView, IndexFilter
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from scipy import io as sio
from sklearn.manifold import TSNE
import math
import os


def show_simple_bar(title, x_axis_label, y_axis_label, source, x, y):
    # Create the blank plot
    result_plot = figure(title=title,
                         x_axis_label=x_axis_label,
                         y_axis_label=y_axis_label,
                         tools=tools_list, tooltips="@%s: @% s" % (x, y))

    result_plot.vbar(x=x, top=y, width=0.5, alpha=0.4, source=source)  # , legend=x, )

    result_plot.xgrid.grid_line_color = None
    result_plot.y_range.start = 0
    return result_plot


def create_sample_scatter(x_data, y_data, source, title='', x_axis_title='', y_axis_title=''):
    result_plot = figure(title=title, tools=tools_list, tooltips=custom_tooltip)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    for label in roman_label:
        index_list = []
        legend_label = ''
        for i in range(len(source.data['style_label'])):
            if source.data['style_label'][i] == label:
                index_list.append(i)
                legend_label = source.data['legend_label'][i]
        view = CDSView(source=source, filters=[IndexFilter(index_list)])
        result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=12,
                            # marker=factor_mark('style_label', markers, roman_label),
                            color=factor_cmap('style_label', 'Category20_16', roman_label),
                            muted_alpha=0.1, view=view,
                            legend_label=legend_label)
    result_plot.legend.click_policy = "hide"

    # highlight x y axes
    # result_plot.renderers.extend([vline, hline])

    return result_plot


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

# input_file_name = 'shape_index_10'
#
#
# if len(sys.argv) >= 2:
#     input_file_name = sys.argv[1]

mat_root_dir = '../mat/20201101'
mat_file_names = os.listdir(mat_root_dir)
N = len(mat_file_names)
cols = round(math.sqrt(N))
print('N: {}, cols: {}'.format(N, cols))

roman_label = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
markers = ['hex', 'triangle', 'circle', 'cross', 'diamond', 'square', 'x', 'inverted_triangle']
output_file('result/20201101_tsne_all.html')

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
# vline = Span(location=0, dimension='height', line_color='black', line_width=2)
# hline = Span(location=0, dimension='width', line_color='black', line_width=2)

grid_list = list()
for mat_name in mat_file_names:
    mat_path = os.path.join(mat_root_dir, mat_name)
    digits = sio.loadmat(mat_path)
    X, labels, styles = digits.get('feature_matrix'), digits.get('label')[0], digits.get('style')
    file_names, indexes = digits.get('relative_file_name'), digits.get('index')[0]
    n_samples, n_features = X.shape

    x_layer = X
    # x_layer = x_layer.reshape(x_layer.shape[0], -1)

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='random', random_state=42)
    X_tsne = tsne.fit_transform(x_layer)
    print("After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter,
                                                                                          x_layer.shape[-1],
                                                                                          X_tsne.shape[-1]))
    data = {'X': X_tsne.T[0],
            'Y': X_tsne.T[1],
            'style_label': [roman_label[labels[i]] for i in range(len(labels))],
            'legend_label': styles,
            'file_name_label': file_names,
            'file_path_label': ["images/" + file_name for file_name in file_names],
            'index_label': indexes,
            }
    sample_source = ColumnDataSource(data=data)
    plot = create_sample_scatter(x_data="X", y_data="Y", source=sample_source,
                                 title=mat_name.replace('.mat', ''), )
    grid_list.append(plot)

p = gridplot(grid_list, ncols=cols, plot_width=800, plot_height=800)
show(p)
