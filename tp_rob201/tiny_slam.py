""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle

import cv2
import numpy as np
import heapq
from matplotlib import pyplot as plt

OCCUPANCY_MAX = +1.0
OCCUPANCY_MIN = -1.0

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map * self.resolution
        y_world = self.y_min_world + y_map * self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(
            np.logical_and(x_px >= 0, x_px < self.x_max_map),
            np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val

    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # * TP4
        # get lidar values, robot referencial
        distances = lidar.get_sensor_values()   # 
        angles = lidar.get_ray_angles()         # angles in radiums

        # get array of bool's where the lidar found obstacules
        isObstacule = distances < lidar.max_range

        # get robot's position, odometer referencial (robot's initial position)
        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]

        # get lidar values with obstacules, odometer referencial
        # distances[isObstacule] only gives the values of distances where isObstacule is true
        # therefore it only calculates for the needed values
        xObs = x_0 + distances[isObstacule] * np.cos(angles[isObstacule] + angle_0)
        yObs = y_0 + distances[isObstacule] * np.sin(angles[isObstacule] + angle_0)

        # get coordenates in the map reference
        xObsMap, yObsMap = self._conv_world_to_map(xObs, yObs)

        # keep only values inside the map
        isValidX = xObsMap < self.x_max_map
        isValidY = yObsMap < self.y_max_map

        xObsMap = xObsMap[isValidX * isValidY]
        yObsMap = yObsMap[isValidX * isValidY]

        isValidX = 0 <= xObsMap
        isValidY = 0 <= yObsMap

        xObsMap = xObsMap[isValidX * isValidY]
        yObsMap = yObsMap[isValidX * isValidY]
        # is possible for a variable be inside and another outside
        # therefore a point will be consider only with both x and y are inside with "and" operation

        # sum the occupancy of obstacules points
        score = np.sum(self.occupancy_map[xObsMap, yObsMap])

        # score between [-360, +360]
        # worst case: all points are not obstacules and they are all equal to OCCUPANCY_MIN
        # best  case: all points are     obstacules and they are all equal to OCCUPANCY_MAX

        # as the lidar has 360 measuares, OCCUPANCY_MAX = +1 (obstacule) and OCCUPANCY_MIN = -1 (empty)
        # worst case: 360*-1 = -360
        # best  case: 360*+1 = +360
        # note that this values change according to the global constants chosen

        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # * TP4

        # initialize reference with not given
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # odometer original reference
        x_odom_ref = odom_pose_ref[0]
        y_odom_ref = odom_pose_ref[1]
        ang_odom_ref = odom_pose_ref[2]

        # robot position in his odometer reference
        x_odom = odom[0]
        y_odom = odom[1]
        ang_odom = odom[2]

        # distance travelled by the robot
        distance = np.sqrt(x_odom**2 + y_odom**2)

        # angle turned by the robot in relation of the odometer origin
        ang_rotation = np.arctan2(y_odom, x_odom)

        # convert absolute map position
        corrected_pose = []

        x_corrected = x_odom_ref + distance * np.cos(ang_rotation + ang_odom_ref)
        y_corrected = y_odom_ref + distance * np.sin(ang_rotation + ang_odom_ref)

        corrected_pose.append(x_corrected)
        corrected_pose.append(y_corrected)
        corrected_pose.append(ang_odom + ang_odom_ref)
        # as angles are measure in relation of it's last reference their sum
        # will be in relation of the world reference

        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # * TP4

        # initialize score with the reference position
        bestScore = self.score(lidar, odom)
        bestRef = odom

        # search for a better score by random variations
        i = 0
        N = 1.25e2
        # try to find a better score in N tries
        # execution with N bigger makes system slower
        while i < N:
            # random value following a gaussien distribution
            # angle is more sensible, use smaller offset
            # offsets should be changed by hand
            offset = []
            offset.append(np.random.normal(0.0, 5))
            offset.append(np.random.normal(0.0, 5))
            offset.append(np.random.normal(0.0, 0.15))
            newRef = bestRef + offset

            # add offset to reference
            odomOffset = self.get_corrected_pose(odom, newRef)
            offsetScore = self.score(lidar, odomOffset)

            # if a new score is found, reset tries
            if offsetScore > bestScore:
                i = 0

                bestScore = offsetScore
                bestRef = newRef
            # keep searching
            else:
                i += 1

        # saving best reference found
        self.odom_pose_ref = bestRef

        return bestScore

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # * TP3
        # get lidar values, robot referencial
        distances = lidar.get_sensor_values()   # 
        angles = lidar.get_ray_angles()         # angles in radiums

        # define threshold for border values
        border = 20

        # get array of bool's where the lidar found obstacules
        isObstacule = distances <= (lidar.max_range - border)

        # get robot's position, odometer referencial (robot's initial position)
        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]

        # get lidar values, odometer referencial
        x = distances * np.cos(angles + angle_0) + x_0
        y = distances * np.sin(angles + angle_0) + y_0

        # increase points values, obstacule
        self.add_map_points(x[isObstacule], y[isObstacule], +0.35)  # modèle simple

        # decrease points values, free path
        for x, y in zip(x, y):
            self.add_map_line(x_0, y_0, x, y, -0.10)

        # set upper and lower limit of point's value
        self.occupancy_map[self.occupancy_map >= OCCUPANCY_MAX] = OCCUPANCY_MAX
        self.occupancy_map[self.occupancy_map <= OCCUPANCY_MIN] = OCCUPANCY_MIN

    def read_map(self, map_csv: str) -> None:
        """
        read Bayesian map from csv file
        map_csv : path to saved map as csv
        """

        try:
            self.occupancy_map = np.genfromtxt(map_csv, delimiter=',')
        except IOError:
            print(f'error: file {map_csv} not found')

        return None


    def get_neighbors(self, current):
        """
        get the 8 neighbors of a point in the map
        current : point in the map
        """

        x, y = current
        neighbors = []

        for i in [-1, 0, +1]:
            for j in [-1, 0, +1]:
                if i == 0 and j == 0:
                    continue

                new_x, new_y = x+i, y+j
                if (new_x < 0 or new_x >= self.x_max_map or
                    new_y < 0 or new_y >= self.y_max_map):
                    continue

                neighbors.append((new_x, new_y))

        return neighbors


    def heuristic(self, a, b):
        # unfolding coordinates
        x_a, y_a = a
        x_b, y_b = b

        # euclidean distance
        return np.sqrt( (x_b - x_a)**2 + (y_b - y_a)**2 )


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """

        # convert from world to map coordinates
        start = self._conv_world_to_map(start[0], start[1])
        goal  = self._conv_world_to_map(goal[0], goal[1])

        # print(f'start: {start} goal: {goal}')

        # heapq initialization
        priority_heap = [(0, start)]
        visited_nodes = set()

        # 
        parent_nodes = {start: None}

        # scores dictionaries
        g_values = {start: 0}
        f_values = {start: self.heuristic(start, goal)}

        # print('plan')
# 
        # loop util heap is empty, meaning: no path found or goal found
        while priority_heap:
            # print('while')
            # pop node with smallest f value
            _, current_node = heapq.heappop(priority_heap)

            # print(f'current_node: {current_node}')

            # if goal was found, reconstruct path and return it
            if current_node == goal:
                path = []

                while current_node is not None:
                    # print('path')
                    path.append(current_node)
                    current_node = parent_nodes[current_node]

                # return reverse path
                return path[::-1]

            # goal not found, store current node as visited and continue searching
            visited_nodes.add(current_node)

            neighbors = self.get_neighbors(current_node)
            # print(f'neighbors: {neighbors}')

            # loop through possible movements
            for neighbor in neighbors:
                # print(f'neighbor: {neighbor}')
                # print('neighbor')

                x_valid = 0 <= neighbor[0] < self.x_max_map
                y_valid = 0 <= neighbor[1] < self.y_max_map
                notObstacule = self.occupancy_map[neighbor[0]][neighbor[1]] < 0.75*OCCUPANCY_MAX

                print(f'{x_valid} {y_valid} {notObstacule}')

                # check if neighbor is inside bounders and is not an obstacle
                if x_valid and y_valid and notObstacule:
                    # compute neighbor's g value
                    tentative_g_value = g_values[current_node] + self.heuristic(current_node, neighbor)

                    # print(f'tentative_g_value: {tentative_g_value}')
                    # print(f'visited_nodes: {visited_nodes}')
                    # print(f'g_values: {g_values.get(neighbor, 0)}')

                    # 
                    if neighbor in visited_nodes and tentative_g_value >= g_values.get(neighbor, 0):
                        continue

                    parent_nodes[neighbor] = current_node
                    g_values[neighbor] = tentative_g_value
                    f_values[neighbor] = tentative_g_value + self.heuristic(neighbor, goal)

                    heapq.heappush(priority_heap, (f_values[neighbor], neighbor))

            # break

        # if heap is empty and no path was found
        return None

        #         # calculate the tentative cost to reach the neighbor
        #         tentative_cost = cost[current_node] + 1

        #         # if neighbor is not in the open set, add it
        #         if neighbor not in cost or tentative_cost < cost[neighbor]:
        #             cost[neighbor] = tentative_cost

        #             f_score = tentative_cost + heuristic(neighbor, goal)

        #             heapq.heappush(open_set, (f_score, neighbor))

        #             path_indices = [goal_index] + path_indices
        #             path_coords = np.unravel_index(path_indices, shape=graph.shape)

        #             path = list(zip(path_coords[0], path_coords[1]))

        #             # path[neighbor] = current_node

        # # exhausted all possibilities, no path avaliable
        # return None


        # self._conv_map_to_world(x_map, y_map)
        # self._conv_world_to_map(x_world, y_world)



        # // A * finds a path from start to goal .
        # // h is the heuristic function . h ( n ) estimates the cost to reach goal from node n .
        # function A_Star ( start , goal , h )
        # // The set of discovered nodes that may need to be ( re -) expanded .
        # // Initially , only the start node is known .
        # // This is usually implemented as a min - heap or priority queue rather than a
        # hash - set .
        # openSet := { start }
        # // For node n , cameFrom [ n ] is the node immediately preceding it on the cheapest
        # path from the start
        # // to n currently known .
        # cameFrom := an empty map
        # // For node n , gScore [ n ] is the cost of the cheapest path from start to n
        # currently known .
        # gScore := map with default value of Infinity
        # gScore [ start ] := 0
        # // For node n , fScore [ n ] := gScore [ n ] + h ( n ) . fScore [ n ] represents our current
        # best guess as to
        # // how cheap a path could be from start to finish if it goes through n .
        # fScore := map with default value of Infinity
        # fScore [ start ] := h ( start )
        # while openSet is not empty
        # // This operation can occur in O ( Log ( N ) ) time if openSet is a min - heap or a
        # priority queue
        # current := the node in openSet having the lowest fScore [] value
        # if current = goal
        # return r e c o n s tr u c t _ p a t h ( cameFrom , current )
        # openSet . Remove ( current )
        # for each neighbor of current
        # // d ( current , neighbor ) is the weight of the edge from current to
        # neighbor
        # // t e n t a t i v e _ g S c o r e is the distance from start to the neighbor through
        # current
        # t e n t a t i v e _ g S c o re := gScore [ current ] + d ( current , neighbor )
        # if t e n t a t i v e _ g S c o r e < gScore [ neighbor ]
        # // This path to neighbor is better than any previous one . Record it
        # !
        # cameFrom [ neighbor ] := current
        # gScore [ neighbor ] := t e n t a t i v e _g S c o r e
        # fScore [ neighbor ] := t e n t a t i v e _g S c o r e + h ( neighbor )
        # if neighbor not in openSet
        # openSet . add ( neighbor )
        # // Open set is empty but goal was never reached
        # return failure

        # TODO for TP5

        path = [start, goal]  # list of poses
        return path


    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T,
                   origin='lower',
                   extent=[
                       self.x_min_world, self.x_max_world, self.y_min_world,
                       self.y_max_world
                   ])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(
            robot_pose[0],
            robot_pose[1],
            delta_x,
            delta_y,
            color='red',
            head_width=5,
            head_length=10,
        )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2,
                        pt1=pt1,
                        pt2=pt2,
                        color=(0, 0, 255),
                        thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T,
                   origin='lower',
                   extent=[
                       self.x_min_world, self.x_max_world, self.y_min_world,
                       self.y_max_world
                   ])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump(
                {
                    'occupancy_map': self.occupancy_map,
                    'resolution': self.resolution,
                    'x_min_world': self.x_min_world,
                    'x_max_world': self.x_max_world,
                    'y_min_world': self.y_min_world,
                    'y_max_world': self.y_max_world
                }, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
