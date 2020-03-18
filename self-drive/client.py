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


def train(dataPath):
    model = Model(dataPath, 128, (7984 // 128), (1996 // 128), 15)
    model.train('C:\\Users\\Teo\\Desktop\\self-driving-agent\\self-drive\\models\\checkpoints', 'model_checkpoint.h5', 'C:\\Users\\Teo\\Desktop\\self-driving-agent\\self-drive\\models', 'model_final.h5')


def collectData(frames):
    world = connect()
    car = CarControl(world)  # create car instance
    car.spawnCar()
    car.attachCamera()
    car.record(frames)
    car.destroy()  # get rid of actors


def main():
    train('C:\\Users\\\Teo\\\Desktop\\\data_10k\\')


if __name__ == '__main__':
    main()
