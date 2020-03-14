import glob
import os
import sys
from CarControl import CarControl

try:
    sys.path.append(glob.glob('C:/Users/Teo/Desktop/self-driving-agent/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    car = CarControl(world) # create car instance
    car.spawnCar()
    car.attachCamera()

    car.record(20) # number of frames (will be changed)

    car.destroy() # get rid of actors

if __name__ == '__main__':
    main()