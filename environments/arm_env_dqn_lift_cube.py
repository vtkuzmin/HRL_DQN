from collections import namedtuple
import numpy as np
import sys
from gym import spaces
import gym
from environments.arm_env_dqn import up_scaler, ArmEnvDQN


class ArmEnvDQN_2(ArmEnvDQN):
    def place_cubes(self, seed=None):
        if seed:
            np.random.seed(seed)
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)

        cubes_left = self._cubes_cnt
        while cubes_left != 0:
            column = np.random.randint(self._size_y)
            for i in np.arange(self._size_x - 1, 0, -1):
                if self._grid[i, column] == 0 and (self._size_x - i) < self._tower_target_size:
                    self._grid[i, column] = 1
                    cubes_left -= 1
                    break

    def __init__(self, size_x, size_y, cubes_cnt, scaling_coeff, episode_max_length, finish_reward, action_minus_reward,
                 tower_target_size, seed=None):

        # checking for grid overflow
        assert cubes_cnt < size_x * size_y, "Cubes overflow the grid"

        self._size_x = size_x
        self._size_y = size_y
        self._cubes_cnt = cubes_cnt
        self._episode_max_length = episode_max_length
        self._finish_reward = finish_reward
        self._action_minus_reward = action_minus_reward
        self._tower_target_size = tower_target_size
        self._scaling_coeff = scaling_coeff
        self.seed = seed

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(size_x * self._scaling_coeff, size_y * self._scaling_coeff, 3))

        self._episode_rewards = []
        self._episode_lengths = []

        _ = self.reset()

    def reset(self):
        self._episode_length = 0
        self._episode_reward = 0
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)
        self._arm_x = 0
        self._arm_y = np.random.randint(self._size_y)
        self._done = False
        self._magnet_toggle = False

        #         cubes_left = self._cubes_cnt
        #         for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
        #             if cubes_left == 0:
        #                 break
        #             cubes_left -= 1
        #             self._grid[x, y] = 1
        self.place_cubes(self.seed)

        self._tower_height = self.get_tower_height()  # инициализируем высоту башни
        self._current_state = self._grid

        return self.get_evidence_for_image_render()

    def get_evidence_for_image_render(self):
        res = np.array(self._grid, copy=True)
        arm_scale = self._scaling_coeff
        res[self._arm_x][self._arm_y] = 2
        res = up_scaler(res, arm_scale)
        for (x, y), value in np.ndenumerate(res):
            if value == 2:
                res[x:x + arm_scale, y:y + arm_scale] = 0
                res[x:x + arm_scale, y + arm_scale // 2] = 2
                res[x + arm_scale - 1, y:y + arm_scale] = 2
                break
        if self._magnet_toggle:
            res[res == 2] = 3

        size_i, size_j = res.shape
        channels = 3

        # Create an empty image
        img = np.zeros((size_i, size_j, channels), dtype=np.uint8)

        # Set the RGB values
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if res[x][y] == 1:
                    img[x][y] = (230, 200, 150)

                if res[x][y] == 2:
                    img[x][y] = (204, 0, 0)

                if res[x][y] == 3:
                    img[x][y] = (51, 153, 255)
        return img

    def ok(self, x, y):
        return 0 <= x < self._grid.shape[0] and 0 <= y < self._grid.shape[1]

    def ok_and_empty(self, x, y):
        return self.ok(x, y) and self._grid[x][y] == 0

    # def grid_to_img(self):
    #     """ Возвращает np.array размера [size_x, size_y] """
    #     grid = np.array(self._grid, copy=True)
    #     grid[self._arm_x, self._arm_y] = 3 - self._magnet_toggle * 1
    #     return grid

    def get_tower_height(self):
        h = 0
        for j in range(self._grid.shape[1]):
            t = 0
            for i in np.arange(self._grid.shape[0] - 1, 0, -1):
                if self._grid[i, j] == 1 and self._grid[i - 1, j] == 0 and (
                                    i + 1 == self._grid.shape[0] or self._grid[i + 1, j] == 1):
                    t = self._grid.shape[0] - i
                    break
            if t > h:
                h = t
        return h

    def step(self, a):

        self._episode_length += 1

        if a in self.MOVE_ACTIONS:
            cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
            cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
            if self._magnet_toggle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1:
                new_arm_x, new_arm_y = self._arm_x + self.MOVE_ACTIONS[a][0], self._arm_y + self.MOVE_ACTIONS[a][1]
                new_cube_x, new_cube_y = new_arm_x + cube_dx, new_arm_y + cube_dy
                self._grid[cube_x][cube_y] = 0
                if self.ok_and_empty(new_arm_x, new_arm_y) and self.ok_and_empty(new_cube_x, new_cube_y):
                    self._arm_x, self._arm_y = new_arm_x, new_arm_y
                    self._grid[new_cube_x][new_cube_y] = 1
                else:
                    self._grid[cube_x][cube_y] = 1
            else:
                new_arm_x, new_arm_y = self._arm_x + self.MOVE_ACTIONS[a][0], self._arm_y + self.MOVE_ACTIONS[a][1]
                if self.ok_and_empty(new_arm_x, new_arm_y):
                    self._arm_x, self._arm_y = new_arm_x, new_arm_y
                else:
                    # cant move, mb -reward
                    pass
        elif a == self.ACTIONS.ON:
            self._magnet_toggle = True
        elif a == self.ACTIONS.OFF:
            cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
            cube_x, cube_y = self._arm_x + cube_dx, self._arm_y + cube_dy
            if self.ok(cube_x, cube_y) and self._grid[cube_x, cube_y] == 1 and self._magnet_toggle:
                new_cube_x, new_cube_y = cube_x + cube_dx, cube_y + cube_dy
                while self.ok_and_empty(new_cube_x, new_cube_y):
                    new_cube_x, new_cube_y = new_cube_x + cube_dx, new_cube_y + cube_dy
                new_cube_x, new_cube_y = new_cube_x - cube_dx, new_cube_y - cube_dy
                self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[
                    new_cube_x, new_cube_y]
                self._magnet_toggle = False

        observation = self._grid
        self._current_state = observation
        reward = self._action_minus_reward
        if a == 0 or a == 2:
            reward += 10 * self._action_minus_reward

        # if self._tower_height < self.get_tower_height():
        #     self._tower_height = self.get_tower_height()
        #     reward += 10
        self._episode_reward += reward

        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        if self._arm_x == 0 and self._grid[1, self._arm_y] == 1 and self._grid[2, self._arm_y] == 0:
            self._done = True
            reward += self._finish_reward
            self._episode_reward += self._finish_reward
            info = True
            self._episode_rewards.append(self._episode_reward)
            self._episode_lengths.append(self._episode_length)
            return self.get_evidence_for_image_render(), reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
            self._episode_rewards.append(self._episode_reward)
            self._episode_lengths.append(self._episode_length)
        return self.get_evidence_for_image_render(), reward, self._done, info

