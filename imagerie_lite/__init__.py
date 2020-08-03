__VERSION__ = '0.0.8'

from imagerie_lite.imagerie_lite import (order_points, remove_lonely_small_objects, biggest_contour,
                                         get_biggest_contour, closest_point, get_corners, warp_perspective,
                                         warp_homography, image_composite_with_mask, combine_two_images_with_mask,
                                         prepare_for_prediction_single, prepare_for_prediction, fill_holes,
                                         remove_smaller_objects, calculate_distance, midpoint, line_intersection)

from imagerie_lite.operations.img import img_as_uint, img_as_float, img_as_bool
