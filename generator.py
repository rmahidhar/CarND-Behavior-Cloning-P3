import pandas
import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os
import cv2
import matplotlib.pyplot as plot
from PIL import Image

class DataGenerator(object):
    def __init__(self, data, image_dir, training_mode=True, batch_size=128):
        self.__data = data
        self.__image_dir = image_dir
        self.__training_mode = training_mode
        self.__cameras = ['left', 'center', 'right']
        self.__cameras_steering_correction = [.25, 0., -.25]
        self.__batch_size = batch_size
        self.__cnt = 0

    @staticmethod
    def brightness_image(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * .30
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return rgb_image

    def __get_random_image(self):
        image_choice = np.random.randint(len(self.__data))
        camera_choice = np.random.randint(len(self.__cameras))
        # reads image in RGB, cv2 return in BGR
        image_filename = self.__image_dir + self.__data[cameras[camera_choice]].values[image_choice].strip()
        image = mpimg.imread(image_filename)
        steering_angle = self.__data.steering.values[image_choice] + cameras_steering_correction[camera_choice]
        return image, steering_angle

    def __add_image(self):
        if self.__cnt < self.__batch_size:
            self.__x[self.__cnt] = self.__image
            self.__y[self.__cnt] = self.__steering_angle
            self.__cnt += 1
            return True
        return False

    def __add_bright_image(self):
        image = np.copy(self.__image)
        image = DataGenerator.brightness_image(image)
        if self.__cnt < self.__batch_size:
            self.__x[self.__cnt] = image
            self.__y[self.__cnt] = self.__steering_angle
            self.__cnt += 1
            return True
        return False

    def __call__(self):
        while True:
            self.__x = np.zeros((self.__batch_size, 160, 320, 3), dtype=np.float32)
            self.__y = np.zeros(self.__batch_size, dtype=np.float32)
            self.__cnt = 0
            while self.__cnt < self.__batch_size:
                self.__image, self.__steering_angle = self.__get_random_image()
                if self.__training_mode:
                    flip = np.random.randint(2)
                    if flip == 0:
                        self.__image = cv2.flip(self.__image, 1)
                        self.__steering_angle = -self.__steering_angle

                    if not self.__add_image():
                        break

                    if not self.__add_bright_image():
                        break
                else:
                     self.__add_image()

            yield(self.__x, self.__y)




