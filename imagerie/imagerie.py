from skimage.morphology import remove_small_objects

from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_fill_holes as sk_binary_fill_holes

from numpy import array as np_array, argsort, where as np_where, vstack, int0, float32, ndarray

from cv2 import (findContours, contourArea, goodFeaturesToTrack, getPerspectiveTransform, findHomography,
                 warpPerspective, RANSAC, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

from PIL.Image import Image, composite, fromarray, open
from PIL.JpegImagePlugin import JpegImageFile

from skimage.io import imread as sk_imread
from skimage.transform import resize as sk_resize


import numpy as np


def order_4_coordinates_clockwise(points: list):
    """ Sorts the 4 (x, y) points clockwise starting from top-left point. """

    points = np_array(points)

    x_sorted = points[argsort(points[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    right_most = right_most[argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    return np_array([tl, tr, br, bl])


def remove_lonely_small_objects(grayscale):
    """ Removes lonely small objects from binary mask, the \"grayscale\" parameter must be a grayscale. """

    binary = np_where(grayscale > 0.1, 1, 0)
    processed = remove_small_objects(binary.astype(bool))

    mask_x, mask_y = np_where(processed == 0)
    grayscale[mask_x, mask_y] = 0

    return grayscale


def biggest_contour(grayscale):
    """ Finds and retrieves the biggest contour """

    contours = findContours(grayscale, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    contour_sizes = [(contourArea(contour), contour) for contour in contours]
    biggest = max(contour_sizes, key=lambda x: x[0])[1]

    return biggest


def get_biggest_contour(contours):
    """ Simply retrieves the biggest contour """

    contour_sizes = [(contourArea(contour), contour) for contour in contours]
    biggest = max(contour_sizes, key=lambda x: x[0])[1]

    return biggest


def closest_point(point: tuple, points):
    """ Returns the closest (x, y) point from a given list of (x, y) points/coordinates. """

    return points[cdist([point], points).argmin()]


def get_corners(grayscale, middle_points=False, centroid=False, max_corners=4, quality_level=0.01, min_distance=15):
    """ Returns the (x, y) coordinates of the 4 corners of a rectangular shaped object from binary mask by default.
    However, you can also calculate the top and bottom middle coordinates by providing \"middle_points=True\".
    And by providint \"centroid=True\", you can get the (x, y) coordinates of the center. """

    corners = goodFeaturesToTrack(grayscale, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corners = int0(corners)

    if corners is None:
        raise Exception('[error][imagerie] Could not detect corners.')

    corners2 = []
    for cr in corners:
        x, y = cr.ravel()
        corners2.append([x, y])

    corners = np_array(corners2)
    corners = order_4_coordinates_clockwise(corners)
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
    """ Performs a warpPerspective() operation and expects the 4 (x, y) coordinates of the source and destination image. """

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

    img = sk_imread(img)
    img = sk_resize(img, shape) / 255.0

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


def binary_fill_holes(img: ndarray):
    """ Fills black holes that reside inside of a binary object (basically a white object in a grayscale image)
    """
    mask = np.logical_not(input)
    tmp = np.zeros(mask.shape, bool)
    inplace = isinstance(output, numpy.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin)
        numpy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1,
                                 origin)
        numpy.logical_not(output, output)
        return output
