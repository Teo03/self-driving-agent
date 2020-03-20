import glob
import os
import sys
import time
from car_control import CarControl
from model import Model

try:
    sys.path.append(glob.glob('C:/Users/Teo/Desktop/self-driving-agent/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def connect():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    world = client.get_world()
    return world


def collectData(frames):
    world = connect()
    car = CarControl(world)

    # set Autopilot to true for collecting data
    car.spawnCar(True)
    car.attachCamera()
    car.record(frames)

    # get rid of actors
    car.destroy()


def train(dataPath, modelsPath, numTrain, numVal, bSize):
    model = Model(dataPath, bSize, (numTrain // bSize), (numVal // bSize), 10)
    model.train(modelsPath)


def drive():
    world = connect()
    car = CarControl(world)

    # set Autopilot to false because we want to control the car
    car.spawnCar(False)
    car.attachCamera()
    car.engage()

    time.sleep(10)
    car.destroy()


def main():
    if sys.argv[1] == 'train':
        train(str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[1] == 'collect':
        collectData(int(sys.argv[2]))
    elif sys.argv[1] == 'drive':
        drive()


if __name__ == '__main__':
    main()
