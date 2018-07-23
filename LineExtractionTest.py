import numpy as np

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage import io
from skimage.feature import canny

import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import measure
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from PIL import Image
from pylsd.lsd import lsd
import math
import csv


def find_contour(img):
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.5)
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
    edges = canny(img, sigma=10, low_threshold=0.05, high_threshold=0.10)
    # Classic straight-line Hough transform
    h, theta, d = hough_line(edges, )
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
    # edges = canny(img, sigma=4)
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
    edges = canny(img, sigma=10, low_threshold=0.10, high_threshold=0.20)

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
    # ax.axis((0, 350, 350, 0))
    plt.show()


def calculate_line_length(line):
    _x = line[2] - line[0]
    _y = line[3] - line[1]
    return math.sqrt(_x ** 2 + _y ** 2)


def calculate_point_line_distance(point, line):
    line_end0, line_end1 = (line[0], line[1]), (line[2], line[3])
    x_diff = line_end1[0] - line_end0[0]
    y_diff = line_end1[1] - line_end0[1]
    num = abs(y_diff * point[0] - x_diff * point[1] + line_end1[0] * line_end0[1] - line_end1[1] * line_end0[0])
    den = math.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / den


def calculate_slope(line):
    line *= 1.0
    _x_diff = line[2] - line[0]
    _y_diff = line[3] - line[1]
    if _x_diff == 0.0:
        return 99999.0
    else:
        return _y_diff / _x_diff


def get_thick_lines(lines):
    slope_offset_threshold = 0.04
    distance_upper_threshold = 22.0
    distance_lower_threshold = 18.0
    thick_lines = []
    for i in range(len(lines) - 1):
        line_0 = lines[i]
        for line_1 in lines[i:]:
            if abs(calculate_slope(line_0) - calculate_slope(line_1)) < slope_offset_threshold:
                distance_list = [
                    calculate_point_line_distance(line_0[0:2], line_1),
                    calculate_point_line_distance(line_0[2:4], line_1),
                    calculate_point_line_distance(line_1[0:2], line_0),
                    calculate_point_line_distance(line_1[2:4], line_0),
                ]
                for distance in distance_list:
                    if distance_lower_threshold <= distance <= distance_upper_threshold:
                        # thick_line_0 = (line_0 + line_1) / 2.0
                        # new_line_1 = [line_1[2], line_1[3], line_1[0], line_1[1], line_1[4]]
                        # thick_line_1 = (line_0 + new_line_1) / 2.0
                        # thick_line_0[4] = distance
                        # thick_line_1[4] = distance
                        # thick_lines.append(thick_line_0
                        #                    if calculate_line_length(thick_line_0) > calculate_line_length(
                        #     thick_line_1)
                        #                    else thick_line_1)
                        thick_lines.append([line_0[0], line_0[1], line_0[2], line_0[3], distance])
                        break
    return np.asarray(thick_lines, np.int32)


def get_correct_position(x, y, img):
    x_min, y_min = 0, 0
    x_max, y_max = img.shape[1], img.shape[0]
    _x, _y = x, y
    if _x < x_min:
        _x = x_min
    if _y < y_min:
        _y = y_min
    if _x > x_max - 1:
        _x = x_max - 1
    if _y > y_max - 1:
        _y = y_max - 1
    return _x, _y


def filter_thick_lines(lines, img, thickness):
    thickness_threshold = 0.9
    filtered_lines = []
    for line in lines:
        slope = calculate_slope(line)
        p_slope = -1 / slope if slope != 0.0 else 99999.0
        center_point = [(line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0]
        pixel_position_list_0 = []
        pixel_position_list_1 = []
        portion = math.sqrt(1.0 / (1 + p_slope ** 2))
        for i in range(thickness):
            _x = int(center_point[0] + (i + 1) * portion)
            _y = int(center_point[1] + (i + 1) * portion * p_slope)
            _x, _y = get_correct_position(_x, _y, img)
            pixel_position_list_0.append((_x, _y))
            _x = int(center_point[0] - (i + 1) * portion)
            _y = int(center_point[1] - (i + 1) * portion * p_slope)
            _x, _y = get_correct_position(_x, _y, img)
            pixel_position_list_1.append((_x, _y))
        pixel_list_0 = [img[pixel[1], pixel[0]] for pixel in pixel_position_list_0]
        pixel_list_1 = [img[pixel[1], pixel[0]] for pixel in pixel_position_list_1]
        aver_0 = 1.0 * sum(pixel_list_0) / len(pixel_list_0)
        aver_1 = 1.0 * sum(pixel_list_1) / len(pixel_list_1)
        if (aver_0 < thickness * thickness_threshold or aver_1 < thickness * thickness_threshold):
            filtered_lines.append(line)
            # filtered_lines.append(
            #     [pixel_position_list_0[0][0],
            #      pixel_position_list_0[0][1],
            #      pixel_position_list_0[-1][0],
            #      pixel_position_list_0[-1][1],
            #      1])
            # filtered_lines.append(
            #     [pixel_position_list_1[0][0],
            #      pixel_position_list_1[0][1],
            #      pixel_position_list_1[-1][0],
            #      pixel_position_list_1[-1][1], 1])
    return np.asarray(filtered_lines, np.int32)


def write_csv(path, lines):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow((["start_x", "start_y", "end_x", "end_y", "width"]))
        writer.writerows(lines)


def extract_segment(path):
    img = Image.open(path)
    gray = np.asarray(img.convert('L'))
    threshold = 96
    thresholdedData = (gray > threshold) * 255
    lines = lsd(thresholdedData)
    fig, ax = plt.subplots()  # figsize=(15, 15))
    ax.imshow(thresholdedData, interpolation='nearest', cmap=plt.cm.gray)
    # thick_lines = get_thick_lines(lines)
    thick_lines = filter_thick_lines(lines, thresholdedData, 20)
    thick_lines = lines
    for i in range(thick_lines.shape[0]):
        p0, p1 = (int(thick_lines[i, 0]), int(thick_lines[i, 1])), (int(thick_lines[i, 2]), int(thick_lines[i, 3]))
        width = thick_lines[i, 4]
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=width / 2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    csv_path = path.split('.')[0] + '.csv'
    write_csv(csv_path, np.ndarray.tolist(thick_lines))


# Constructing test image
path = r'C:\Users\bunny\Desktop\pompeii\test\test3.png'
image = io.imread(path, as_grey=True)

# classic_hough_line(image)
# find_contour(image)
# canny_test(image)
# prob_hough_line(image)
# detect_corner(image)
extract_segment(path)
