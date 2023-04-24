"""
Robot controller definition
Complete controller including SLAM, planning, path following

origin in orange part:
    x+: frontwards
    y+: leftwards
    theta: zero pointing frontwards
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid
from control import potential_field_control


SCORE_MIN = +150


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

        # self.tiny_slam.compute()
        # self.tiny_slam.update_map(self.lidar(), self.odometer_values())


        # if self.counter == 0:
        #     self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        # else:
        bestScore = self.tiny_slam.localise((self.lidar()), self.odometer_values())

        if bestScore > SCORE_MIN:
            # self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
            # self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
            self.tiny_slam.update_map(self.lidar(), self.tiny_slam.get_corrected_pose(self.odometer_values()))
        else:
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        if self.counter % 1 == 0:
            self.tiny_slam.display2(self.odometer_values())

        # Compute new command speed to perform obstacle avoidance
        isReactive = False

        if isReactive is True:
            command = reactive_obst_avoid(self.lidar())
        else:
            command = potential_field_control(self.lidar(), self.odometer_values(), np.array([-100.0, -400.0, 0]))

        # studying score behavior to determine it's threshold
        # print(f"score: {bestScore:+.4e}")
        # self.array_score.append(bestScore)
        # if self.counter % 24 == 0:
        #     print(f'score mean: {np.mean(self.array_score):4.4f} {np.max(self.array_score):4.4f} {np.min(self.array_score):4.4f}')
        #     self.array_score = []
        # this is only needed during setup, afterwards the lines can be commented

        return command
