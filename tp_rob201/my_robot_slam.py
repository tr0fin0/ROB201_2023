"""
Robot controller definition
Complete controller including SLAM, planning, path following

origin in orange part:
    x+: frontwards
    y+: leftwards
    theta: zero pointing frontwards
"""
import numpy as np
import datetime

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid
from control import potential_field_control


SCORE_MIN = +50


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        # array of calculates scores used for debug
        self.array_score = []

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1
        # print(f'{self.counter}')

        # declare default command
        command = {"forward": 0, "rotation": 0}

        # change behavior for A* algorithm
        cartography_limit = 32500

        if self.counter < cartography_limit:
            # explore map to create a cartography

            # initialize occupancy map
            if self.counter <= 30:
                # use robot odometer without any correction
                self.tiny_slam.update_map(self.lidar(), self.odometer_values())

            else:
                # search best reference correction
                score = self.tiny_slam.localise((self.lidar()), self.odometer_values())

                # update occupancy map only with corrected reference is good enough
                if score > SCORE_MIN:
                    self.tiny_slam.update_map(self.lidar(), self.odometer_values()+self.corrected_pose)

            # compute new command speed to perform obstacle avoidance
            isReactive = True

            if isReactive is True:
                command = reactive_obst_avoid(self.lidar())
            else:
                command = potential_field_control(self.lidar(), self.odometer_values(), np.array([-100.0, -400.0, 0]))

        elif self.counter == cartography_limit:
            # save occupancy map
            occupancy_map = self.tiny_slam.occupancy_map

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = './tp_rob201/maps/'

            np.savetxt(f'{save_path}occupancy_map_{now}.csv', occupancy_map, delimiter=',')

        else:
            print(f'A*')

        # display occupancy map within a certain frequency
        if self.counter % 1 == 0:
            self.tiny_slam.display2(self.odometer_values())


        return command