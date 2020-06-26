from scipy.ndimage import binary_fill_holes
from models.segmenter import Segmenter
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


mask = Image.open('mask_risky.jpg').convert('L')
mask.show()

gray = np.array(mask)

inter = cv2.morphologyEx(gray, cv2.MORPH_ELLIPSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)

out = np.zeros(gray.shape, np.uint8)
cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
out = cv2.bitwise_and(gray, out)



mask = out

print('\n')
print(mask.dtype)
print('\n')

Image.fromarray(mask).show()
