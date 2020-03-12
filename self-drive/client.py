import glob
import os
import sys
import random
import time

try:
    sys.path.append(glob.glob('C:/Users/Teo/Desktop/carla/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]

    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)

    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)
    vehicle.set_autopilot(True)

if __name__ == '__main__':
    main()