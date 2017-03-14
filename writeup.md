#Behavior Cloning

The goal of this project is to train an end-to-end convolutional neural network model using keras that would let a car drive itself around the track in a simulator. 

* Data Collection
* Data Augmentation
* Model
* Result

[//]: # (Image References)

[image1]: ./images/data2.png "Training Samples Histogram 2"
[image2]: ./images/cameras.png "Random Sample"
[image3]: ./images/flipped.png "Flipped"
[image4]: ./images/bright.png "Bright"

###Data Collection

I tried collecting the data by driving the car in the simulator. However i didn't succeeded collecting better data compared to the udacity data.

The driving data has driving log that has pointer to the location of the images (frame captured by the simulator) and steering angle, throttle, speed, etc at the time of image captured by the simulator.  

###Data Augmentation

![alt text][image1]

The histogram plots the steering angles of the udacity data samples. The driving data has 8036 samples, and removing 20% of samples for validation data left us with 6429 samples for training the model. These fewer samples are not enough for better generalization of the model and requires augmentation tricks to extend the data.

Each sample has frames from 3 camera positions: left, center and right. Although only central camera is used while driving, we can still use left and right cameras data during training after applying steering angle correction. This increase the number of examples by a factor of 3.

![alt text][image2]

Flipping half of the frames horizontally and change the sign of the steering angle, increase the number of examples by a factor of 2.

![alt text][image3]

The challenging track is bright at many places and hence augmenting bright image might help in generalizing the model better for driving in the challenging track.

![alt text][image4]

The python generator class geneartes batch of images with the above mentioned augmentation techiniques. This generator object is passed to train the keras model.

```python
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
```

#Model

I started with the model(based on the Nvidia paper) presented by david silver in the behavior cloning videos, and then by tweaking cropping layer and removing the bottom 3 convolutional layer the model sucessfully drives the car on track 1 and with some success on track 2. I can clearly observe the difference did by the normalization in this model, removing the normalization layer resulted in the car becoming unstable on the tracks. Dropout is not used in the model as it didn't showed any improvement in the model.

Layer (type) | Output Shape | Params | Connected to
-------------|--------------|-------|--------------
lambda_1 (Lambda) | (None, 160, 320, 3) |  0  | input_1[0][0]
cropping2d_1 (Cropping2D) |  (None, 60, 300, 3) |   0 |  lambda_1[0][0]
convolution2d_1 (Convolution2D) | (None, 28, 148, 24)| 1824|      cropping2d_1[0][0]
convolution2d_2 (Convolution2D) | (None, 12, 72, 36) | 21636  |    convolution2d_1[0][0]
convolution2d_3 (Convolution2D)|  (None, 4, 34, 48)  | 43248  |    convolution2d_2[0][0]
flatten_1 (Flatten)  | (None, 6528)  | 0 |  convolution2d_3[0][0]
dense_1 (Dense) | (None, 100)    | 652900 |   flatten_1[0][0]
dense_2 (Dense)  | (None, 50)   |  5050   |     dense_1[0][0]
dense_3 (Dense) | (None, 10)    |  510       |  dense_2[0][0]
dense_4 (Dense)| (None, 1)   |   11      |    dense_3[0][0]

	Total params: 725,179
	Trainable params: 725,179
	Non-trainable params: 0

The model is trained using adam optimizer, mean square error loss function with learning rate of 0.0001 for 25 epochs.

#Result
The car drives endlessly on the track it was trained, but on the challange track it drives half way and crashed at a steep turn while descending.

<p align="center">
  <img src="images/training.gif" alt="Driving autonomously on track 2"/>
</p>


I'm unsucessful to make the car drive continoulsy on the challenging track.  

<p align="center">
  <img src="images/challange.gif" alt="Driving autonomously on track 2"/>
</p>
