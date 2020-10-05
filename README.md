Imagerie
========
> Lightweight python package grouping together useful computer vision functions and operations

This package was initially created for myself to extract some of the most useful important functions,  
from packages like `scipy`, `skimage`, etc...  

This was done in order to expose all these very useful computer vision functions,  
and be able to use them in a **Serverless** environment.

Documentation
-------------

Prerequisites
-------------
This package only supports `python3` today.

Only tested on `python3.6` and `python3.7`

Installation
------------
Install using pip `pip install imagerie`.

Functions
---------

##### [`imagerie.get_rotation(pt1, pt2)`](#)
Returns the rotation value in degrees between 2 (x,y) points.

##### [`imagerie.order_points(points)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L20)
You've guessed it, this function simply sorts a list of 4 [x, y] coordinates in a clockwise manner, starting from the top-left.

##### [`imagerie.remove_lonely_small_objects(grayscale)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L235)
This function removes small white objects from a binary mask.  

##### [`imagerie.remove_smaller_objects(grayscale)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L247)
Finds all contours and removes them all except the biggest one.

##### [`imagerie.biggest_contour(grayscale)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L39)
Finds and retrieves the biggest contour from a grayscale image.

##### [`imagerie.get_biggest_contour(contours)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L47)
Simply retrieves and returns the biggest contour from a given list of contours.

##### [`imagerie.closest_point(point: tuple, points)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L62)
Returns the closest (x, y) point from a given list of (x, y) points/coordinates.

##### [`imagerie_lite.midpoint(ptA, ptB)`](#)
Calculates the X,Y middle points from provided 2 points.

##### [`imagerie_lite.line_intersection(line1: tuple, line2: tuple)`](#)
Returns the intersection point between two lines.

##### [`imagerie.get_corners(grayscale, middle_points=False, centroid=False, max_corners=4, quality_level=0.01, min_distance=15)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L79)
Returns the (x, y) coordinates of the 4 corners of a rectangular shaped object from binary mask by default.
However, you can also calculate the top and bottom middle coordinates by providing `middle_points=True`.
And by providint `centroid=True`, you can get the (x, y) coordinates of the center.

##### [`imagerie.warp_perspective(image, src_pts: list, dst_pts: list)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L134)
Performs a `cv2.warpPerspective()` operation and expects 2 lists of (x, y) corner points of the source 
and destination image.  

##### [`imagerie.warp_homography(image, src_pts: list, dst_pts: list, method=cv2.RANSAC, reproj_threshold=5.0)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L149)
Performs a `cv2.warpPerspective()` operation after `cv2.findHomography()`.

##### [`imagerie.image_composite_with_mask(to_add: PIL.Image.Image, destination: PIL.Image.Image, mask: PIL.Image.Image)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L164)
Combines the `to_add` and `destination` images, `to_add` image will be added on top of `destination` image
and only the white area from the `mask` image will be retained from `to_add` image.

##### [`imagerie.combine_two_images_with_mask(background_img, foreground_img, mask)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L174)
Combines the images with the help of the provided mask.
Note that only the white area of the mask will be selected from the `foreground_img`.

##### [`imagerie.prepare_for_prediction_single(img, shape=(768, 768), as_array=True)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L208)
Loads and resizes a single image to a given shape (default: 768, 768) and returns it by default as a numpy array.

##### [`imagerie.prepare_for_prediction(imgs, shape=(768, 768))`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L222)
Does the same thing as `imagerie.prepare_for_prediction_single` but for multiple images.

##### [`imagerie.fill_holes(gray_img: ndarray, min=200, max=255)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/imagerie.py#L261)
Fills black pixel holes that reside inside of a binary object.  

> `min` represents the `thresh` used in `cv2.threshold(gray, thresh=200, ...)`.  
>
> `max` represents the `maxval` used in `cv2.threshold(gray, maxval=255, ...)`.

##### [`imagerie.img_as_uint(img)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/operations/img.py#L226)
Converts image to 16 bit unsigned integer format.

##### [`imagerie.img_as_float(img)`](https://github.com/ibragim64/imagerie/blob/ec9087742d165ecf116856574988874b39274325/imagerie/operations/img.py#L233)
Convert an image to floating point format.

Credits
-------
 - [Ibragim Abubakarov](https://www.ibragim.fr) <[ibragim.ai95@gmail.com](mailto:ibragim.ai95@gmail.com)>