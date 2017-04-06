import cv2
import numpy as np
from copy import deepcopy
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import math
from scipy.ndimage.morphology import distance_transform_bf

def binary_image(image, show=False):
    
    #binarization
    bin_image = np.zeros((image.shape[0], image.shape[1]))
    t = 1.6 * threshold_otsu(image)
    mask = image.sum(axis=2)
    bin_image[mask > t] = 255
    bin_image[mask <= t] = 0
    
    #delete small objects
    label_objects, nb_labels = ndi.label(bin_image)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 10000
    mask_sizes[0] = 0
    bin_image = mask_sizes[label_objects]
    
    if show:
        plt.imshow(bin_image)
    
    return bin_image

def area_(x, y, z):
    xy = x - y
    xz = x - z
    return np.abs(xy[0] * xz[1] - xy[1] * xz[0])

def preproc(image, bin_image):
    img = deepcopy(image)
    contours, hierarchy = cv2.findContours(bin_image,cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_NONE)
    max_area = -1
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area=area
            ci=i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt,returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    all_defects = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]
        start = cnt[s][0]
        end = cnt[e][0]
        far = cnt[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b *c)) * 180 / (np.pi)
        s = area_(far, start, end)
        if (angle <= 95) and (s > 4000):
            all_defects.append([tuple(start), tuple(far), tuple(end)])
    return all_defects

def filtering(image, bin_image, all_defects):
    img = deepcopy(image)
    if (len(all_defects) == 4):
        return np.ones(len(all_defects)).astype('bool')
    d_transform = distance_transform_bf(bin_image)
    x_center, y_center = np.where(d_transform == d_transform.max())
    x_center = x_center[0]
    y_center = y_center[0]
    
    vectors = np.zeros((len(all_defects), 2))
    for i, defects in enumerate(all_defects):
        far = defects[1]
        vector = np.array([far[0] - y_center, far[1] - x_center])
        vector = vector / np.linalg.norm(vector)
        vectors[i, :] = vector
    mean_vector = vectors.mean(axis=0)
    
    is_tip_valley = np.zeros(len(all_defects)).astype('bool')
    
    for i in range(vectors.shape[0]):
        scalar_mult = (mean_vector * vectors[i]).sum()
        is_tip_valley[i] = (scalar_mult > 0.0)
    return (is_tip_valley)

def features(image, is_tip_valley, all_defects):
    img = deepcopy(image)
    feature_vector = []
    defects = np.array(all_defects)[is_tip_valley]
    if (defects.shape[0] != 4):
        raise ValueError('Array must be shape[0] == 4')
    
    for i in range(defects.shape[0]):
        if i + 1 != defects.shape[0]:
            dist = np.linalg.norm(defects[i][2] - defects[i + 1][0])
        else:
            dist = np.linalg.norm(defects[i][2] - defects[0][0])
        if (dist > 30.0):
            break
    
    if i != defects.shape[0] - 1:
        defects = np.roll(defects, -6 * (i + 1))
    
    for i in range(defects.shape[0] - 1):
        mean_point = (defects[i][2] + defects[i + 1][0]) // 2
        defects[i][2] = mean_point
        defects[i + 1][0] = mean_point
    
    for i in range(defects.shape[0]):
        a = np.linalg.norm(defects[i][0] - defects[i][1])
        b = np.linalg.norm(defects[i][2] - defects[i][1])
        feature_vector.append(a)
        feature_vector.append(b)
        cv2.line(img, tuple(defects[i][0]), 
                 tuple(defects[i][1]), [0, 255, 0], 2)
        cv2.line(img, tuple(defects[i][2]), 
                 tuple(defects[i][1]), [0, 255, 0], 2)
        cv2.circle(img, tuple(defects[i][0]), 
                   1, [0, 255, 0], 5)
        cv2.circle(img, tuple(defects[i][1]), 
                   1, [0,0,255], 5)
        cv2.circle(img, tuple(defects[i][2]), 
                   1, [0, 255, 0], 5)
    if (feature_vector[1] < feature_vector[-2]):
        feature_vector = feature_vector[::-1]
    feature_vector = np.array(feature_vector)   
    
    return (img, feature_vector)