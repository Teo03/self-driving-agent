import glob
import os
import sys
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


def train(dataPath, modelsPath, numTrain, numVal):
    model = Model(dataPath, 128, (numTrain // 128), (numVal // 128), 20)
    model.train(modelsPath)


def collectData(frames):
    world = connect()
    car = CarControl(world)

    car.spawnCar()
    car.attachCamera()
    car.record(frames)

    # get rid of actors
    car.destroy()


def main():
    if sys.argv[1] == 'train':
        train(str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == 'collect':
        collectData(int(sys.argv[2]))


if __name__ == '__main__':
    main()
