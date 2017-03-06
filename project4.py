#%%
import numpy as np
import cv2
from glob import glob
from PIL import Image
from ipywidgets import interact, fixed, IntSlider
from matplotlib import pyplot as plt
from itertools import islice

#%% Helper functions

def show_images(images):
    fig = plt.figure()
    for i in range(len(images)):
        img = images[i]
        a = fig.add_subplot(1, len(images), i + 1)
        plt.imshow(img, cmap=(None if len(img.shape) == 3 else 'gray'))
        #plt.axis('off')
    plt.show()

def show_rgb(img):
    show_images([img])

def show_hsv(images):
    show_images(cv2.cvtColor(img, cv2.COLOR_HSV2RGB) for img in images)

def show_gray(images):
    show_images(images)

#%% Camera calibration

chess_shape = (9, 6)
objpoints = []
imgpoints = []
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
for path in glob('camera_cal/*'):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chess_shape, None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

for path in islice(glob('camera_cal/*'), 2):
    img = cv2.imread(path)
    dst = undistort(img)
    show_images([img, dst])


#%% Perspective correction
src = np.float32([
        [253, 685],
        [572, 465],
        [709, 465],
        [1052, 685]])

dst = np.float32([
        [500, 750],
        [500, 300],
        [700, 300],
        [700, 750]])

M = cv2.getPerspectiveTransform(src, dst)

def get_road(img):
    img_size = (img.shape[1], img.shape[0])
    # we mostly are interested in upscaling and CUBIC seems to be quite good
    # http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_CUBIC)

for i in islice(glob('test_images3/*'), 2):
    img = np.asarray(Image.open(i))
    road = get_road(img)
    show_images([img, road])


#%% Road markings detection
def find_lane_yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = np.zeros_like(h)
    mask[(h > 15) & (h <= 31)] = 1
    return mask

def find_lane_gray_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray)
    mask[gray >= 225] = 1
    return mask

class Line():
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels
        self._poly = None

    def poly(self):
        if not self._poly:
            residuals, rank, singular_values, rcond, pre = np.polyfit(y, x, 2, full=True)
            self._poly = np.poly1d(residuals, variable='y')

        return self._poly


    def could_contain_pixel(self, p):
        max_distance = 2
        if (self.minx - max_distance < p[1] < self.maxx + max_distance and
            abs(p[0] - self.pixels[-1][0]) < 2):
            return True
        return False

    def add_pixel(self, p):
        self.pixels.append(p)
        self.minx = min(self.minx, p[0])
        self.maxx = max(self.maxx, p[0])


def find_bright_lanes(img):
    g_min = np.min(img)
    g_max = np.max(img)
    gray = (img - g_min) * (1. / (g_max - g_min))

    line_width = 11
    # -1, ..., -1, 1, ..., 1, -1, ..., -1
    kernel = np.concatenate((np.repeat(-1, line_width / 2),
                             np.repeat(1, line_width),
                             np.repeat(-1, line_width / 2)))

    kernel = kernel / np.sum(np.abs(kernel))

    conv = cv2.filter2D(gray, -1, kernel.reshape(1, -1))
    c_max = np.max(conv)
    conv[conv < c_max * 0.2] = 0

    lines = []

    from scipy.ndimage import measurements, morphology
    labels, nbr_objects = measurements.label(conv)

    for label in range(1, nbr_objects + 1):
        lines.append(Line(label, np.where(labels == label)))

    print("Number of objects: {}".format(nbr_objects))

    #show_images([conv])

    #print("Found %s line candidates" % len(lines))

    marked = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    #print(lines)

    for line in lines:
        marked[line.pixels] = [255, 200, 200]
        f = line.poly()

        ys = range(0, img.shape[0])
        print(f)
        #xs = line.poly(ys)

    show_images([marked])



def find_lanes(img):
    show_images([img])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    yellow = np.copy(h)
    #yellow = cv2.inRange(h, 18, 30)
    yellow[(h < 18) | (h > 30)] = 0
    lane_colors = np.maximum(gray, yellow)
    find_bright_lanes(lane_colors)


for i in islice(glob('test_images3/*'), 4):
    img = np.array(Image.open(i))
    #show_images([img])
    find_lanes(get_road(img))


