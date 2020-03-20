import os
import cv2
import random
import numpy as np
from imgaug import augmenters as img_aug


class Preprocess:
    def __init__(self, image_path):
        self.imagePath = image_path
        
    # TODO: change back to private
    def readImg(self, image_name):
        # get image as array of values
        path = self.imagePath + image_name
        image = cv2.imread(path)
        return image

    @staticmethod
    def __pan(image):
        pan = img_aug.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        return image

    @staticmethod
    def __zoom(image):
        zoom = img_aug.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
        return image

    @staticmethod
    def __adjustBrightness(image):
        brightness = img_aug.Multiply((0.7, 1.3))
        image = brightness.augment_image(image)
        return image

    @staticmethod
    def __blur(image):
        kernel_size = random.randint(1, 5)
        image = cv2.blur(image, (kernel_size, kernel_size))
        return image

    def __augment(self, image, steering_angle):
        if np.random.rand() < 0.5:
            image = self.__pan(image)
        if np.random.rand() < 0.5:
            image = self.__zoom(image)
        if np.random.rand() < 0.5:
            image = self.__blur(image)
        if np.random.rand() < 0.5:
            image = self.__adjustBrightness(image)
        return image, steering_angle

    @staticmethod
    def preprocess(image):
        # based on nvidia paper
        height, _, _ = image.shape
        image = image[int(height / 2):, :, :]  # removes top half
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image = cv2.resize(image, (200, 66))
        return image / 255 # normalize

    def image_data_generator(self, image_names, steering_angles, batch_size, is_training):
        while True:
            batch_images = []
            batch_steering_angles = []

            for i in range(batch_size):
                # get a random image from array of image indexes
                random_index = random.choice(image_names.index.values)

                image = self.readImg(image_names[random_index])
                steering_angle = steering_angles[random_index]

                if is_training:
                    image, steering_angle = self.__augment(image, steering_angle)

                image = self.preprocess(image)
                batch_images.append(image)
                batch_steering_angles.append(steering_angle)

            yield np.asarray(batch_images), np.asarray(batch_steering_angles)