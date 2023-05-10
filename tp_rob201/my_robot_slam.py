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
from control import wall_follow
from control import potential_attraction


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
        self.tiny_slam = TinySlam(x_min= -self._size_area[0],
                                  x_max= +self._size_area[0],
                                  y_min= -self._size_area[1],
                                  y_max= +self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.command_choice = 'wall_follow'
        self.save_map = False
        self.explore = True
        self.explore_counter_limit = 1000

        self.path = None
        self.path_return = None

    def control(self):
        """
        Main control function executed at each time step
        """

        # ! setup
        # default command
        command = {"forward": 0, "rotation": 0}


        # save robot origin
        if self.counter == 0:
            self.corrected_pose = self.odometer_values()

            if self.explore is False:
                # self.tiny_slam.read_map('./tp_rob201/maps/occupancy_map_2023-05-03_09-45-43.csv')
                self.tiny_slam.read_map('./tp_rob201/maps/occupancy_map_2023-05-03_10-01-46.csv')


        # ! exploration
        if self.explore is True:
            # explore map to create a cartography
            if self.counter < self.explore_counter_limit:

                # initialize occupancy map
                if self.counter <= 30:
                    # use robot odometer without any correction
                    self.tiny_slam.update_map(self.lidar(), self.odometer_values())

                else:
                    # search best reference correction
                    score = self.tiny_slam.localise((self.lidar()), self.odometer_values())

                    # update occupancy map only with corrected reference is good enough
                    if score > SCORE_MIN:
                        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

                # * command choice
                if self.counter > 75:

                    match self.command_choice:
                        case 'reactive':
                            command = reactive_obst_avoid(self.lidar())

                        case 'potential_field':
                            command = potential_field_control(self.lidar(), self.odometer_values(), np.array([-100.0, -400.0, 0]))

                        case 'wall_follow':
                            command = wall_follow(self.lidar())

                        case _:
                            command = {"forward": 0, "rotation": 0}


            elif self.counter == self.explore_counter_limit:
                if self.save_map is True:
                    # save occupancy map
                    occupancy_map = self.tiny_slam.occupancy_map

                    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path = './tp_rob201/maps/'

                    np.savetxt(f'{save_path}occupancy_map_{now}.csv', occupancy_map, delimiter=',')


        # ! planinning
        # from the actual position plan the best way to the start position
        if self.path is None and self.counter > self.explore_counter_limit:
            goal = self.corrected_pose[:2].tolist()
            start = self.odometer_values()[:2].tolist()

            self.path = self.tiny_slam.plan(start, goal)
            self.path_return = self.path

        if self.counter <= tmp_counter:
            command = reactive_obst_avoid(self.lidar())


        # ! update 
        # display occupancy map within a certain frequency
        if self.counter % 1 == 0:
            # self.tiny_slam.display(self.odometer_values())
            self.tiny_slam.display2(self.odometer_values())

        # increase counter
        self.counter += 1
        # print(f'{self.counter}')

        return command