# from scipy.ndimage import binary_fill_holes
# from models.segmenter import Segmenter
from PIL import Image
import cv2
import imagerie
import numpy as np


payload = [
    {
        'image': 'audi.jpg',
        'background': None
    }
]

# segmenter = Segmenter((768, 768), '.')
# mask = segmenter.get_mask('audi.jpg')

# mask.save('mask.jpg')

# mask = np.array(mask)

# mask = imagerie.binary_fill_holes(mask)
#
# cv2.imwrite('result.jpg', mask * 255)

#
# mask = Image.open('mask_risky.jpg').convert('L')
# mask.show()
#
# gray = np.array(mask)

gray = cv2.imread('mask_noisy.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Mask', gray)
cv2.waitKey(5000)
cv2.destroyAllWindows()

mask = imagerie.remove_smaller_objects(gray)

cv2.imshow('Result', mask)
cv2.waitKey(5000)
cv2.destroyAllWindows()
