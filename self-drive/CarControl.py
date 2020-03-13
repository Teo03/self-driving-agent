import glob
import os
import sys
import random
import time
import numpy as np
import cv2

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

    def spawnCar(self):
        #spawns and returns car actor
        
        blueprint_library = self.world.get_blueprint_library()

        car_bp = blueprint_library.filter('model3')[0] # spawn tesla model 3 :)

        if car_bp.has_attribute('color'):
            car_bp.set_attribute('color', '204, 0, 0') # tesla red color
        transform = random.choice(self.world.get_map().get_spawn_points())

        self.car = self.world.spawn_actor(car_bp, transform)
        print('created %s' % self.car.type_id)
        self.car.set_autopilot(True)
    
    def attachCamera(self):
        # spawns and attaches camera sensor

        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb') # it's actually rgba(4 channel)

        cam_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        cam_bp.set_attribute('fov', '110') # field of view

        # time in seconds between sensor captures
        cam_bp.set_attribute('sensor_tick', '1.5')

        # attach the camera
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # spawn the camera
        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.car)
        self.camera.listen(lambda data: self.__cameraFeed(data))
        
    def __cameraFeed(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4)) # 4 channel - rgba
        i3 = i2[:, :, :3] # 3 channel rgb
        cv2.imshow("", i3)
        cv2.waitKey(1)
        return i3/255.0

    def destroyCar(self):
        # use this instead of actor_list
        
        self.car.destroy()
        self.camera.destroy()
        print('destroyed vehicle and sensors')
