from imagerie_lite.operations.morphology import remove_small_objects, binary_fill_holes

from numpy import array as np_array, argsort, where as np_where, vstack, int0, float32, ndarray

from cv2 import (findContours, contourArea, goodFeaturesToTrack, getPerspectiveTransform, findHomography, bitwise_and,
                 warpPerspective, imread, resize, threshold, morphologyEx, getStructuringElement, drawContours)

from cv2 import (RANSAC, RETR_EXTERNAL, CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, THRESH_BINARY, FILLED,
                 MORPH_CLOSE, MORPH_ELLIPSE)

from PIL.Image import Image, composite, fromarray, open
from PIL.JpegImagePlugin import JpegImageFile

from imagerie_lite.operations.img import img_as_uint, img_as_float

import numpy as np
import math


def order_points(points: ndarray):
    """ Sorts the 4 (x, y) points clockwise starting from top-left point. """

    x_sorted = points[argsort(points[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # right_most = right_most[argsort(right_most[:, 1]), :]
    D = calculate_distance(tl[np.newaxis], right_most)
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    return np_array([tl, tr, br, bl], dtype='float32')


def biggest_contour(grayscale):
    """ Finds and retrieves the biggest contour """

    contours, _ = findContours(grayscale, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

    return max(contours, key=contourArea)


def get_biggest_contour(contours):
    """ Simply retrieves the biggest contour """

    return max(contours, key=contourArea)


def calculate_distance(pt1, pt2):
    """ Calculates the spacial distance between 2 (x,y) points """

    if type(pt1) is tuple and type(pt2) is tuple:
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return dist

    result = None
    pt1_type = type(pt1)
    pt2_type = type(pt2)
    if pt2_type is list or pt2_type is ndarray:
        result = []

        if pt1_type is list or pt1_type is ndarray:
            pt1 = pt1[0]
        else:
            pt1 = pt1

        x1, y1 = pt1

        for pt in pt2:
            if pt2_type is list:
                x2, y2 = pt
            else:
                x2, y2 = pt.ravel()

            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            result.append(dist)

    return np.array(result, dtype='float32')


def closest_point(point: tuple, points):
    """ Returns the closest (x, y) point from a given list of (x, y) points/coordinates. """

    distances = []
    for x, y in points:
        dist = calculate_distance(point, (x, y))
        distances.append(dist)

    return points[np.array(distances).argmin()]


def midpoint(ptA, ptB):
    """ Calculates X,Y middle points from provided 2 points. """

    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def line_intersection(line1: tuple, line2: tuple):
    """ Returns the intersection point between two lines. """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('[imagerie] lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return int(x), int(y)


def get_corners(grayscale, middle_points=False, centroid=False, max_corners=4, quality_level=0.01, min_distance=15):
    """ Returns the (x, y) coordinates of the 4 corners of a rectangular shaped object from binary mask by default.
    However, you can also calculate the top and bottom middle coordinates by providing \"middle_points=True\".
    And by providing \"centroid=True\", you can get the (x, y) coordinates of the center. """

    corners = goodFeaturesToTrack(grayscale, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corners = int0(corners)

    if corners is None:
        raise Exception('[error][imagerie_lite] Could not detect corners.')

    corners2 = []
    for cr in corners:
        x, y = cr.ravel()
        corners2.append([x, y])

    corners = np_array(corners2)
    corners = order_points(corners)
    corners = int0(corners)

    c1 = tuple(corners[0])
    c2 = tuple(corners[1])
    c3 = tuple(corners[2])
    c4 = tuple(corners[3])
    
    corners = [c1, c2, c3, c4]

    x = [p[0] for p in corners]
    y = [p[1] for p in corners]
    centroid = (sum(x) / len(corners), sum(y) / len(corners))

    if not middle_points:
        if not centroid:
            return corners
        else:
            return [corners, centroid]

    contours, _ = findContours(grayscale, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    cnt = get_biggest_contour(contours)

    centroid_top_approx = (int(centroid[0]), int(centroid[1]) - 2)
    centroid_bottom_approx = (int(centroid[0]), int(centroid[1]) + 5)

    centroid_top = closest_point(centroid_top_approx, vstack(cnt).squeeze())
    centroid_bottom = closest_point(centroid_bottom_approx, vstack(cnt).squeeze())

    centroid_top = (centroid[0], centroid_top[1])
    centroid_bottom = (centroid[0], centroid_bottom[1])

    if not centroid:
        return int0([c1, centroid_top, c2, c3, centroid_bottom, c4])
    else:
        return [int0([c1, centroid_top, c2, c3, centroid_bottom, c4]), centroid]


def warp_perspective(image, src_pts, dst_pts, shape: tuple):
    """ Performs a warpPerspective() operation and expects the 4 (x, y) coordinates of the source and destination
    image. """

    width, height = shape

    src_pts = float32(src_pts)
    dst_pts = float32(dst_pts)

    h = getPerspectiveTransform(src_pts, dst_pts)

    res = warpPerspective(image, h, (width, height))

    return res


def warp_homography(image, src_pts, dst_pts, shape: tuple, method=RANSAC, reproj_threshold=5.0):
    """ Performs a warpPerspective() operation after findHomography(). """

    width, height = shape

    src_pts = float32(src_pts)
    dst_pts = float32(dst_pts)

    h, _ = findHomography(src_pts, dst_pts, method, reproj_threshold)

    res = warpPerspective(image, h, (width, height))

    return res


def image_composite_with_mask(to_add: Image, destination: Image, mask: Image) -> Image:
    """ Combines the `to_add` and `destination` images, `to_add` image will be added on top of `destination` image
     and only the white area from the `mask` image will be retained from `to_add` image. """

    if mask.mode != 'L':
        mask = mask.convert('L')

    return composite(to_add, destination, mask=mask)


def combine_two_images_with_mask(background_img, foreground_img, mask):
    """ Selects and pastes the content from "foreground_img" to "background_img" with the help of the provided mask.
    """

    if type(background_img) is str:
        background_img = open(background_img)

    if type(background_img) is ndarray:
        background_img = fromarray(background_img)

    if type(background_img) is not Image and type(background_img) is not JpegImageFile:
        raise Exception(f'Type of "background_img" must be one of these types [{Image}, {JpegImageFile}, {ndarray}, str]. "{type(background_img)}" given.')

    if type(foreground_img) is str:
        foreground_img = open(foreground_img)

    if type(foreground_img) is ndarray:
        foreground_img = fromarray(foreground_img)

    if type(foreground_img) is not Image and type(foreground_img) is not JpegImageFile:
        raise Exception(f'Type of "foreground_img" must be one of these types [{Image}, {JpegImageFile}, {ndarray}, str]. "{type(foreground_img)}" given.')

    if type(mask) is str:
        mask = open(mask, 'L')

    if type(mask) is ndarray:
        mask = fromarray(mask).convert('L')

    if type(mask) is not Image and type(mask) is not JpegImageFile:
        raise Exception(f'Type of "mask" must be one of these types [{Image}, {JpegImageFile}, {ndarray}, str]. "{type(mask)}" given.')

    return composite(foreground_img, background_img, mask=mask)


def prepare_for_prediction_single(img: str, shape=(768, 768), as_array=True):
    """ Loads and resizes the image to given shape (default: 768, 768) and returns as a numpy array.
    """

    img = imread(img)
    img = img_as_float(resize(img, shape)) / 255.0

    out = img
    if as_array:
        out = np_array([out])

    return out


def prepare_for_prediction(imgs, shape=(768, 768)):
    """ Loads and resizes each image in "imgs" to a given (default: 768, 768) shape and returns the result as a numpy array.
    """

    out = []
    for img in imgs:
        _img = prepare_for_prediction_single(img, shape=shape, as_array=False)

        out.append(_img)

    return np_array(out)


def remove_lonely_small_objects(grayscale):
    """ Removes lonely small objects from binary mask, the \"grayscale\" parameter must be a grayscale. """

    binary = np_where(grayscale > 0.1, 1, 0)
    processed = remove_small_objects(binary.astype(bool))

    mask_x, mask_y = np_where(processed == 0)
    grayscale[mask_x, mask_y] = 0

    return grayscale


def remove_smaller_objects(grayscale):
    """ Removes all objects from binary mask except the biggest one. """

    inter = morphologyEx(grayscale, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, (5, 5)))
    cnts, _ = findContours(inter, RETR_EXTERNAL, CHAIN_APPROX_NONE)
    cnt = max(cnts, key=contourArea)

    out = np.zeros(grayscale.shape, np.uint8)
    drawContours(out, [cnt], -1, 255, FILLED)
    out = bitwise_and(grayscale, out)

    return out


def fill_holes(gray: ndarray, min=200, max=255):
    """  """

    _, thresh = threshold(gray, min, max, THRESH_BINARY)
    gray = binary_fill_holes(thresh)
    gray = img_as_uint(gray)

    return gray


def translate_image(img, x_shift: int, y_shift: int):
    """ Translates image from a given x and y values. """

    a = 1
    b = 0
    c = x_shift
    d = 0
    e = 1
    f = y_shift

    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
