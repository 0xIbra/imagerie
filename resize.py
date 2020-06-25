from imagerie.transform.sk_resize import resize
import cv2


img = cv2.imread('audi.jpg')

img = resize(img, (768, 768)) / 255.0

cv2.imwrite('result.jpg', img)
