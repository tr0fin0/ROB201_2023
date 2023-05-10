# ROB201 - Navegation Robotique

## Description
This git repository *tp_rob201* is the starting point for your work during the ROB201 project/course. The online, up-to-date version of this code is available at: [GitHub](https://github.com/emmanuel-battesti/tp_rob201).

This amis to be a study in robotics navegation exploring the most common and relevant algorithms that could be used in the market. In our case two windows will be shown:
 - **world:** simulation;
 - **map:** cartography;

example of **world** and **map**:
![alt text](/tp_rob201/images/robot_begin.png)

## Installation
### Dependences
To start working on your ROB201 project, you will have to install this repository (which will install required dependencies). The [INSTALL.md](INSTALL.md) file contains installation tips for Ubuntu and Windows. The code will work also on MacOS, with minimal adaptation of the Ubuntu installation instructions.

### Git
As the code was developed using [Git](https://git-scm.com/) as distributed version control system and the project was stored in [GitHub](https://github.com/tr0fin0/ROB201_2023).

## Roadmap
here some explanations about the general functioning of the code presenting the main decisions that drove the project pointing out what worked, what didn't work and how it could be improved.


### Setup
first of all in `my_robot_slam.py` some variables were added to change the overall behavior of the project:
 - `(bool) self.explore`: indicating if the robot should explore or load a saved map, `True` as default;
 - `(bool) self.save_map`: indicating if the map will be saved by the end of the exploration, , `False` as default;
 - `(int) self.explore_counter_limit`: indicating number of iterations for exploration, `1000` as default;
 - `(str) self.command_choice`:  that choose what method used for exploration, `'wall_follow'` as default;

#### `read_map`
if `self.explore = False` a pre-explored map will be loaded with the following functions:
```python
    ...
    # save robot origin
    if self.counter == 0:
        self.corrected_pose = self.odometer_values()

        if self.explore is False:
            # self.tiny_slam.read_map('./tp_rob201/maps/occupancy_map_2023-05-03_09-45-43.csv')
            self.tiny_slam.read_map('./tp_rob201/maps/occupancy_map_2023-05-03_10-01-46.csv')
    ...


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
```
during the development of the A* algorithm, the presence of those functions allowed a significant reduction in the test time and increased the efficiency for code development.

csv format was chosen because it was a familiar tool even though it may not be the most efficient way to perform this operation.

#### `save_map`
at the end of each exploration the map obtained can be saved if desired using the code below:
```python
    elif self.counter == self.explore_counter_limit:
        if self.save_map is True:
            # save occupancy map
            occupancy_map = self.tiny_slam.occupancy_map

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = './tp_rob201/maps/'

            np.savetxt(f'{save_path}occupancy_map_{now}.csv', occupancy_map, delimiter=',')
```

### Exploration
robot will explore the world and will save it's map as it passes.
#### initialization map
during the first 50 interactions of the code, the robot will not move and will build a basic map. after that, during it's navigation, it's odometry will suffer variations that must be corrected in order for the map to be coherent along the route. when some information of the map is given the `localise` function  can vary the coordinates position of the robot's odometry searching for a new benchmark where most of the estimated obstacle points are actually obstacles as follows:
```python
    if self.counter <= 50:
        # use robot odometer without any correction
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

    else:
        # search best reference correction
        score = self.tiny_slam.localise(self.lidar(), self.odometer_values())

        # update occupancy map only with corrected reference is good enough
        if score > SCORE_MIN:
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())
```
the decision of a minimum score value will depend on how the occupation map values were defined and a more detailed explanation is presented in the code.

#### exploration method
after the initialisation period of the map, the exploration of the world begins. Throughout the development of the project, some navigation algorithms were proposed to study the robot's capacity to explore in an efficient and fast way without causing great distortions in the map like the following:
```python
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
```

the extra algorithm of wall follow was implemented and it is the recommended one for exploration because it presents a low interference on the map and an excellent ability to overcome the obstacles of the proposed world.

it's weakest point however is his origin. if the robot is started very close to the central island it will remain trapped in a loop in this region and will not be able to discover new parts of the map.


#### command: `wall_follow`
As a base, the control considers a constant ground speed and subsequently its operation is based on 2 principles independently:
 - **correction of it's angle**, based on two independent references:
   - angle in relation to the right;
![alt text](/tp_rob201/images/wall_angle.png)
   - distance in relation to the wall;
![alt text](/tp_rob201/images/wall_distance.png)
 - **correction of it's velocity**, if the total angle correction was not significant, thus avoiding sudden movements that cause map deviations even with the localise correction mechanism implemented.

 
moreover, if he is not close to a wall he will simply follow his home direction until a wall is found.

the gains and thresholds used in the code were manually obtained by performing several tests, which was tiring and inefficient, the use of machine learning algorithms could be an effective alternative to find an ideal fit for the variables. a stable version is shown below:
```python
    # basic command
    forward = 0.025
    rotation = 0

    # first the algorithm will correct the robot position, in a rather slow background
    # forward command and then, when the robot is well positioned in space, it will
    # increase it's velocity so it can explorer quickly

    index_right = 90    # index on the distances array of the right side measure
    clearance_wall = 15 # minimal clearance


    # ! search wall
    if min_distance < 2.5 * clearance_wall:
        # wall found, rotate until align right side

        #* correct right side alignment
        error_angle = 1.00 * (min_index - index_right)/360
        if abs(error_angle) >= 0.025:
            # error_angle of 0.042 implies +-15 degrees deviation
            # error_angle of 0.025 implies +- 9 degrees deviation
            rotation += +1.00 * error_angle

        #* correct wall clearance
        error_distance = -0.125 * (min_distance - clearance_wall)/clearance_wall
        if abs(error_distance) >= 0.020:
            rotation += 1.00 * error_distance

    else:
        # ! no wall found, explore
        forward += min_distance / max(distances)


    # ! increase speed if angle is stable
    if abs(rotation) <= 0.020:
        forward += 0.045
```
roughly speaking, the proposed wall following algorithm is not the most efficient as it has a low speed therefore taking a long time to make a complete turn around the map, between 12 and 15 minutes.
### Planinning
after a certain amount of iterations, the robot will stop and trace the shortest path between its current position after exploration and its origin position.

the recommended algorithm was the A* that allows to discover the closest path through successive iterations based on a heuristic and a priority queue. it can be contextualized that A* would be an algorithm like Dijkstra's that evaluates the points in a specific order.

as the algorithm is complex and extensive its particularities will not be presented.
```python

        # ! A* algorithm
        # convert from world to map coordinates
        start = self._conv_world_to_map(start[0], start[1])
        goal  = self._conv_world_to_map(goal[0], goal[1])
```
at the end of its execution, the following path will be added to the map so that we can visualize the path that should be taken:

![alt text](/tp_rob201/images/map_return.png)




#### expand obstacules:
one of the main extensions made to this algorithm was the expansion of obstacles so that the robot does not collide with walls.

as the robot is not punctual, the most efficient path could not be too close to the obstacles. thus, before the algorithm the following code was executed:
```python
        # ! expand obstacules
        # car is not pontual, a clearance distance from the wall is need
        occupancy_map_expanded = self.occupancy_map.copy()

        # define minimal wall clearance in map dimensions
        wall_clearance = 8
        # wall_clearance = 17

        # check every point in the map
        for x in range(self.x_max_map):
            for y in range(self.y_max_map):
                # if this point is not unknown and is not free
                if (self.occupancy_map[x][y] != OCCUPANCY_MIN and
                    self.occupancy_map[x][y] != 0):

                    for i in range(wall_clearance):
                        for j in range(wall_clearance):
                            # check if points are inside map
                            if (x+i < self.x_max_map and x-i >= 0 and 
                                y+j < self.y_max_map and y-j >= 0):
                                # expand all four graph quadrants
                                occupancy_map_expanded[x+i][y+j] = OCCUPANCY_MAX
                                occupancy_map_expanded[x+i][y-j] = OCCUPANCY_MAX
                                occupancy_map_expanded[x-i][y-j] = OCCUPANCY_MAX
                                occupancy_map_expanded[x-i][y+j] = OCCUPANCY_MAX
```
essentially it will go through all the points of the matrix and, where there are points that are not free, represented by occupancy_min, and that are not unknown, represented by 0, and will make all the neighbors in a radius of 8 pixels away to be also considered as obstacles.

as we can see in the image of the path traced the line at a distance from the walls which guarantees that the robot will not collide if it follows that line.

#### show path
during the map update a development problem was the use of different coordinates between map and world and the change in the sign of y during these transformations caused by the construction of the plot that was performed.

as the cv2.flip(...) function is used the y coordinates are always opposite which complicates the development and understanding of the code generates several snippets of corrections.
```python
        ...
        img = cv2.flip(self.occupancy_map.T, 0)
        ...


        # add return path
        if path_original is not None:
            color = (255, 255, 255)

            # correction for printing
            path_corrected = []
            for x, y in path_original:
                x_corr, y_corr = self._conv_map_to_world(x,y)

                path_corrected.append((self._conv_world_to_map(x_corr,-y_corr)))

            for node in range(len(path_corrected)-1):
                cv2.line(img2, path_corrected[node], path_corrected[node+1], color, 1)
```

### Returning
the return of the robot was the most complex part of the development because, for reasons that unfortunately could not be discovered during development.

after having a path to be followed, the idea was to break the path into several points and use an attraction control to make the robot follow one point after another. as the expansion of obstacles guarantees that the robot will be able to follow the proposed path, it would be enough to apply it to the attraction.

however, even starting from the first point of the path to return to the origin, this point was always considered directly as the origin and meant that the reference could not be updated to the other available points, causing the robot to be attracted to places outside the path proposed and consequently there was no guarantee that they were free and caused collisions.
```python

    # ! return
    # until arrive at the goal
    if self.path_return:

        # get current position
        x_pos, y_pos, t_pos = self.tiny_slam.get_corrected_pose(self.odometer_values())
        position = (x_pos, y_pos, t_pos)

        # get next goal
        x_map, y_map = self.path_return[0]
        x_goal, y_goal = self.tiny_slam._conv_map_to_world(x_map, y_map)
        goal = (x_goal, y_goal)

        distance_goal = np.sqrt((x_pos - x_goal)**2 + (y_pos - y_goal)**2)


        # if is already close enough, remove closest nodes
        if distance_goal < 15:
            self.path_return = self.path_return[10:]

        command = potential_attraction(np.array(position), goal)
```


#### command: `potential_attractive`
n any case the attraction works as it should just the goal setting which was not satisfactory and the problem cannot be identified. what everything would indicate would be a reference problem probably caused by the way the coordinates of the world and the map are not directly related, to necessary modifications that probably would not have been correctly made throughout the code and cause the difference and consequent malfunction of the code.
```python
    # * TP2
    # ! attractive potential
    att_const = 1
    att_difference = goal[:2] - position[:2]
    atr_distance = np.sqrt(att_difference[0]**2 + att_difference[1]**2)

    attraction = [0, 0]
    if atr_distance > DST_MIN:
        attraction = att_const * att_difference / atr_distance


    # ! resulting potential
    magnitude_attraction = np.sqrt(attraction[0]**2 + attraction[1]**2)

    if magnitude_attraction == 0:
        command = {"forward": 0, "rotation": 0}

    else:
        potential = attraction / magnitude_attraction
        magnitude = np.sqrt(potential[0]**2 + potential[1]**2)
        angle = (np.arctan2(potential[1], potential[0]) - position[2]) / (2*np.pi)

        forward  = 0.065 * magnitude
        rotation = 1.000 * angle

        # limiting command values
        forward_limited  = max(min(forward,  +1), -1)
        rotation_limited = max(min(rotation, +1), -1)

        command = {"forward": forward_limited, "rotation": rotation_limited}

    return command
```



## Author and Acknowledments
- Guilherme TROFINO:
  - [![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/guilherme-trofino/)
  - [![GitHub](https://i.stack.imgur.com/tskMh.png) GitHub](https://github.com/tr0fin0)


We greatly appreciate our [ROB201](https://perso.ensta-paris.fr/~manzaner/Cours/ROB201/) teachers at [ENSTA](https://www.ensta-paris.fr/):
- Emmanuel Battesti

More information in the [**Place-Bot** GitHub repository](https://github.com/emmanuel-battesti/place-bot). It will be installed automatically with the above procedure, but it is strongly recommended to read the [*Place-Bot* documentation](https://github.com/emmanuel-battesti/place-bot#readme).

