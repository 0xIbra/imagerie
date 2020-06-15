# Imagerie
> Python module grouping together useful computer vision functions and operations

This module was initially created for myself for use in my notebooks to stop re-writing the same code and wasting time.

## Functions 
##### `imagerie.order_4_coordinates_clockwise(points: list)`
You've guessed it, this function simply sorts a list of 4 [x, y] coordinates in a clockwise manner, starting from
the top-left.  

##### `imagerie.remove_lonely_small_objects(grayscale)`
This function removes small white objects from a binary mask.  

##### `imagerie.biggest_contour(grayscale)`
Finds and retrieves the biggest contour from a grayscale image.  

##### `imagerie.get_biggest_contour(contours)`
Simply retrieves and returns the biggest contour from a given list of contours.  

##### `imagerie.closest_point(point: tuple, points)`
Returns the closest (x, y) point from a given list of (x, y) points/coordinates  

##### `imagerie.get_corners(grayscale, middle_points=False, centroid=False, max_corners=4, quality_level=0.01, min_distance=15)`
Returns the (x, y) coordinates of the 4 corners of a rectangular shaped object from binary mask by default.
However, you can also calculate the top and bottom middle coordinates by providing `middle_points=True`.
And by providint `centroid=True`, you can get the (x, y) coordinates of the center.  

##### `imagerie.warp_perspective(image, src_pts: list, dst_pts: list)`
Performs a `cv2.warpPerspective()` operation and expects 2 lists of (x, y) corner points of the source 
and destination image.  

##### `imagerie.warp_homography(image, src_pts: list, dst_pts: list, method=cv2.RANSAC, reproj_threshold=5.0)`
Performs a `cv2.warpPerspective()` operation after `cv2.findHomography()`.  

##### `imagerie.image_composite_with_mask(to_add: PIL.Image.Image, destination: PIL.Image.Image, mask: PIL.Image.Image)`
Combines the `to_add` and `destination` images, `to_add` image will be added on top of `destination` image
and only the white area from the `mask` image will be retained from `to_add` image.

#### `imagerie.combine_two_images_with_mask(background_img, foreground_img, mask)`
Combines the images with the help of the provided mask.
Note that only the white area of the mask will be selected from the `foreground_img`.

#### `imagerie.prepare_for_prediction_single(img, shape=(768, 768), as_array=True)`
Loads and resizes a single image to a given shape (default: 768, 768) and returns it by default as a numpy array.

#### `imagerie.prepare_for_prediction(imgs, shape=(768, 768))`
Does the same thing as `imagerie.prepare_for_prediction_single` but for multiple images.  

#### `imagerie.binary_fill_holes(img)`
Fills black pixel holes that reside inside of a binary object.
