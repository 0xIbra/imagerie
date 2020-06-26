""" Keras """
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

""" Numpy """
import numpy as np

""" Pillow """
from PIL import Image, ImageFilter, ImageOps

""" OpenCV """
from cv2 import findContours, threshold, RETR_TREE, THRESH_BINARY, CHAIN_APPROX_SIMPLE, cvtColor, imread, COLOR_BGR2RGB
import cv2

from imagerie.operations.img import img_as_uint, img_as_float

""" Built-ins """
import tempfile
import uuid
import gc
import os


class Segmenter:

    def __init__(self, shape, output_path=None):
        self.__shape = shape
        self.__output_path = output_path
        self.__model = self.__load_model()
        self.__initialize()

    def __initialize(self):
        self.__tempdir = tempfile.gettempdir()
        if self.__output_path is None:
            self.__output_path = self.__tempdir

        if not os.path.isdir(self.__output_path):
            os.makedirs(self.__output_path)

    def segment(self, images):
        # To be predicted by the model
        imgs = []

        # Array containing dicts of resized images and original shape for each image
        prepared_images = []

        for index, item in enumerate(images):
            prepared = self.__prepare_img(item)
            imgs.append(prepared['resized_img'])
            prepared_images.append(prepared)

        imgs = np.array(imgs)

        y_predicted = self.__model.predict(imgs)

        for index, prediction in enumerate(y_predicted):
            name = '{}.jpg'.format(uuid.uuid4())
            save_path = os.path.join(self.__tempdir, name)
            prepared_images[index]['mask_path'] = save_path

            width, height = self.__shape
            predicted_mask = Image.fromarray(img_as_uint(prediction.reshape(width, height))).convert('L')

            height, width = prepared_images[index]['original_shape']
            predicted_mask = predicted_mask.resize((width, height))

            add_shadow = True
            shadow_x = None
            shadow_y = None
            shadow_spread = None

            if 'add_shadow' in prepared_images[index]:
                add_shadow = prepared_images[index]['add_shadow']

            if 'shadow_x' in prepared_images[index]:
                shadow_x = prepared_images[index]['shadow_x']

            if 'shadow_y' in prepared_images[index]:
                shadow_y = prepared_images[index]['shadow_y']

            if 'shadow_spread' in prepared_images[index]:
                shadow_spread = prepared_images[index]['shadow_spread']

            segmented = self.__apply_mask(
                prepared_images[index]['path'],
                predicted_mask,
                background=prepared_images[index]['background'],
                add_shadow=add_shadow,
                shadow_x=shadow_x,
                shadow_y=shadow_y,
                shadow_spread=shadow_spread
            )

            output_path = os.path.join(self.__output_path, prepared_images[index]['name'])
            segmented.save(output_path)

            prepared_images[index]['segmented'] = output_path

            del prepared_images[index]['resized_img']
            del predicted_mask

        # self.__clean(prepared_images)

        gc.collect()

        return prepared_images

    def get_mask(self, img_path):
        imgs = []
        prepared = self.__prepare_img({'image': img_path, 'background': None})
        imgs.append(prepared['resized_img'])

        imgs = np.array(imgs)

        predicted = self.__model.predict(imgs)
        predicted = predicted[0]
        predicted = predicted.reshape(self.__shape)

        predicted = Image.fromarray(img_as_uint(predicted)).convert('L')

        height, width = prepared['original_shape']
        predicted = predicted.resize((width, height))

        return predicted

    def __apply_mask(self, img_path, mask, background=None, add_shadow=True, shadow_x=None, shadow_y=None,
                     shadow_spread=None):
        original = Image.open(img_path)
        if background is None:
            background = Image.new('RGB', original.size, (255, 255, 255))
        else:
            background = Image.open(background).resize(original.size)

        if add_shadow:
            width, height = original.size

            shadow_mask = mask.copy()
            shadow_mask_cnt = np.array(shadow_mask)

            ret, thresh = threshold(shadow_mask_cnt, 70, 255, THRESH_BINARY)

            contours, _ = findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = contours[0]

                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                print('Left most:', leftmost)
                print('Right most:', rightmost)
                print('Top most:', topmost)
                print('Bottom most:', bottommost)

                default_shadow_x = -(round(height * 0.01))
                default_shadow_y = -(round(width * 0.04))

                if leftmost[1] > rightmost[1]:
                    default_shadow_x = -(round(height * 0.01))
                    print('\n')
                    print('Shadow goes to the right')
                    print('\n')
                elif leftmost[1] < rightmost[1]:
                    default_shadow_x = (round(height * 0.01))
                    print('\n')
                    print('Shadow goes to the left')
                    print('\n')
                elif leftmost[1] == rightmost[1]:
                    pass
                else:
                    pass

                if shadow_x is not None and type(shadow_x) is int:
                    default_shadow_x = -shadow_x

                if shadow_y is not None and type(shadow_y) is int:
                    default_shadow_y = -shadow_y

                shadow_mask = Segmenter.translate_image(shadow_mask, default_shadow_x, default_shadow_y)

                shadow_mask = ImageOps.invert(shadow_mask)
                default_shadow_spread = width * 0.021
                print('DEFAULT SPREAD  : ', default_shadow_spread)
                if shadow_spread is not None and type(shadow_spread) is int:
                    default_shadow_spread = shadow_spread

                shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(default_shadow_spread))
                shadow_mask = ImageOps.invert(shadow_mask)

                shadow_bg = Image.new('L', original.size, 25)
                background = Image.composite(shadow_bg, background, shadow_mask)

                del shadow_mask
                del shadow_bg

        final = Image.composite(original, background, mask)

        return final

    def __load_model(self):
        f = open('models/segmentation/structure.json', 'r')
        json = f.read()
        f.close()

        # model = init_segmentation_model()
        model = model_from_json(json)

        model.load_weights('models/segmentation/weights.h5')

        optimizer = Adam(lr=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def __prepare_img(self, image):
        original_img = cvtColor(imread(image['image']), COLOR_BGR2RGB)
        height, width, channels = original_img.shape
        original_shape = (height, width)

        resized_img = img_as_float(cv2.resize(original_img, self.__shape)) / 255.0

        result = {
            'image': image['image'],
            'background': image['background'],
            'name': '{}_{}'.format(str(uuid.uuid4()), os.path.basename(image['image'])),
            'path': image['image'],
            'resized_img': resized_img,
            'original_shape': original_shape
        }

        if 'background' in image.keys():
            result['background'] = image['background']

        if 'content-type' in image.keys():
            result['content-type'] = image['content-type']

        if 'name' in image.keys():
            result['filename'] = image['name']

        if 'add_shadow' in image:
            add_shadow_type = type(image['add_shadow'])
            if add_shadow_type is int or bool:
                result['add_shadow'] = bool(image['add_shadow'])

        if 'shadow_spread' in image:
            if type(image['shadow_spread']) is int:
                result['shadow_spread'] = image['shadow_spread']

        if 'shadow_x' in image:
            if type(image['shadow_x'] is int):
                result['shadow_x'] = image['shadow_x']

        if 'shadow_y' in image:
            if type(image['shadow_y']) is int:
                result['shadow_y'] = image['shadow_y']

        return result

    def __clean(self, prepared_images):
        for item in prepared_images:
            if os.path.isfile(item['mask_path']):
                os.remove(item['mask_path'])

            if os.path.isfile(item['path']):
                os.remove(item['path'])

            if item['background'] is not None and os.path.isfile(item['background']):
                os.remove(item['background'])

    def get_model(self):
        return self.__model

    @staticmethod
    def translate_image(img, x_shift: int, y_shift: int):
        a = 1
        b = 0
        c = x_shift
        d = 0
        e = 1
        f = y_shift

        return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
