# # from scipy.ndimage import binary_fill_holes
# from models.segmenter import Segmenter
# from PIL import Image
# import cv2
# import imagerie_lite
# import numpy as np
#
#
# payload = [
#     {
#         'image': 'audi.jpg',
#         'background': None
#     }
# ]
#
# segmenter = Segmenter((768, 768), '.')
#
# segmenter.segment(payload)


from imutils.perspective import order_points
from imagerie_lite import order_points as imagerie_order_points
import numpy as np


points = [
    [290, 193],
    [293, 129],
    [102, 90],
    [105, 203]
]

expected = order_points(np.array(points))
result = imagerie_order_points(np.array(points))

print(expected, '\n\n\n', result)


# def order_points_old(pts):
#     pts = np.array(pts)
#     rect = np.zeros((4, 2), dtype='float32')
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#
#     return rect
#
#
# result = order_points_old(points)
# print(expected, '\n\n', result)
