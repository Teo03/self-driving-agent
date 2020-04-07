import glob
import os
import sys
import time

import wandb

from car_control import CarControl
from model import Model

try:
    sys.path.append(glob.glob('./dist/carla-*%d.%d-%s.egg' % (
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


def collectDataAuto(seconds):
    world = connect()
    car = CarControl(world)

    car.spawnCar(True)
    car.attachCamera(None)
    car.record()

    time.sleep(seconds)
    car.destroy()


def train(dataPath, modelsPath, savedModelPath):
    wandb.init(project='self-driving-agent')
    model = Model(dataPath, 100, 300, 200, 50)
    if savedModelPath == 'new':
        model.train(modelsPath, None)
    else:
        model.train(modelsPath, savedModelPath)


def drive(seconds):
    world = connect()
    car = CarControl(world)

    car.spawnCar(False)
    car.attachCamera(None)
    car.engage()

    time.sleep(seconds)
    car.destroy()


def main():
    if sys.argv[1] == 'train':
        train(str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
    elif sys.argv[1] == 'collect':
        collectDataAuto(int(sys.argv[2]))
    elif sys.argv[1] == 'drive':
        drive(int(sys.argv[2]))


if __name__ == '__main__':
    main()
