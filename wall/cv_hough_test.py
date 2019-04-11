import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\bunny\Desktop\pompeii\test\test3.png'
img = cv2.imread(path)


def classic_hough(image):
    global gray, edges, lines, x1, y1, x2, y2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 500, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imwrite('houghlines3.jpg',img)
    im = plt.imshow(image)
    plt.show()


def probabilistic_hough(image):
    global gray, edges, lines, x1, y1, x2, y2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 96, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    #edges = cv2.Canny(im_bw, 100, 200, apertureSize=3)

    minLineLength = 50
    maxLineGap = 20
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            # plt.plot((x1, x2), (y2, y2))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # cv2.imwrite('houghlines5.jpg', image)
    im = plt.imshow(image)
    plt.show()


def canny(image):
    edges = cv2.Canny(img, 200, 500)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


# classic_hough(img)
probabilistic_hough(img)
# canny(img)
