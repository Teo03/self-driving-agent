import glob
import os
import sys

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from image_preprocess import Preprocess

try:
    sys.path.append(glob.glob('C:/Users/Teo/Desktop/self-driving-agent/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IMG_HEIGHT = 480
IMG_WIDTH = 640


class CarControl:

    def __init__(self, world):
        self.world = world
        self.car = None
        self.camera = None
        self.df = pd.DataFrame({'imageName': [], 'steeringAngle': []})
        self.model = load_model('./models/model_final.h5')

    def spawnCar(self, auto):
        # spawns and returns car actor
        
        blueprint_library = self.world.get_blueprint_library()

        car_bp = blueprint_library.filter('model3')[0]  # spawn tesla model 3 :)

        if car_bp.has_attribute('color'):
            car_bp.set_attribute('color', '204, 0, 0') # tesla red color
        transform = self.world.get_map().get_spawn_points()[4]

        self.car = self.world.spawn_actor(car_bp, transform)
        print('created %s' % self.car.type_id)
        self.car.set_autopilot(auto)

    def attachCamera(self, manualCtrCar):
        # spawns and attaches camera sensor

        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')  # it's actually RGBA (4 channel)

        cam_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        cam_bp.set_attribute('fov', '90') # field of view

        # time in seconds between sensor captures
        cam_bp.set_attribute('sensor_tick', '0.1')

        # attach the camera
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # added for manual driving
        if self.car is None:
            self.car = manualCtrCar

        # spawn the camera
        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.car)

    def engage(self):
        self.camera.listen(lambda image: self.__getLiveFeed(image.raw_data))

    def __getLiveFeed(self, raw_img):
        pr = Preprocess('')
        img = np.array(raw_img).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        image = img[:, :, :3]
        final_img = pr.preprocess(image)
        self.__predictAngle(final_img)

    def __predictAngle(self, img):
        predictedAngle = self.model.predict(img.reshape(1, 66, 200, 3))
        print(str(predictedAngle[0][0]))
        self.car.apply_control(carla.VehicleControl(throttle=0.4, steer=float(predictedAngle[0][0])))

    def record(self):
        # record car actions
        self.camera.listen(lambda image: self.__save(image, self.car.get_control()))

    def __save(self, image, control):
        # save the data and the png's

        if self.car.is_at_traffic_light():
            traffic_light = self.car.get_traffic_light()
            if traffic_light.get_state() != carla.TrafficLightState.Green:
                traffic_light.set_state(carla.TrafficLightState.Green)

        path = '../generated_data/image_data/%s.png' % image.frame
        image.save_to_disk(path)

        steer = control.steer
        self.__createRow(path, steer)

    def __createRow(self, image_path, steer):
        # creates a row in the csv file

        image_name = image_path.rsplit('/', 1)[-1]  # get the name without the folder path
        row = [image_name, steer]
        self.df.loc[len(self.df)] = row

        print('saved row: ' + image_name + '/' + str(steer))
        self.df.to_csv("../generated_data/data.csv", index=False)

    def destroy(self):
        if self.car is not None:
            self.car.destroy()
        if self.camera is not None:
            self.camera.destroy()
        exit(0)
        print('destroying actors')
