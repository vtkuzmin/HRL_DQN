from collections import namedtuple
import numpy as np
import sys
from gym import spaces
import gym
from environments.env_core import CoreEnv


def up_scaler(grid, up_size):
    res = np.zeros(shape=np.asarray(np.shape(grid)) * up_size)
    for (x, y), value in np.ndenumerate(grid):
        res[x * up_size:x * up_size + up_size, y * up_size:y * up_size + up_size] = grid[x][y]
    return res


class ArmEnvDQN(CoreEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    ACTIONS = namedtuple("ACTIONS", ["LEFT", "UP", "RIGHT", "DOWN", "ON", "OFF", ])(
        LEFT=0,
        UP=1,
        RIGHT=2,
        DOWN=3,
        ON=4,
        OFF=5,
    )

    MOVE_ACTIONS = {
        ACTIONS.UP: [-1, 0],
        ACTIONS.LEFT: [0, -1],
        ACTIONS.DOWN: [1, 0],
        ACTIONS.RIGHT: [0, 1],
    }

    def __init__(self, size_x, size_y, cubes_cnt, scaling_coeff, episode_max_length, finish_reward, action_minus_reward,
                 tower_target_size):

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
        self._arm_y = 0
        self._done = False
        self._magnet_toggle = False

        cubes_left = self._cubes_cnt
        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if cubes_left == 0:
                break
            cubes_left -= 1
            self._grid[x, y] = 1

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

    def step(self, a, isoption=False):

        if not isoption:
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

        self._episode_reward += reward

        info = None
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        if self.get_tower_height() == self._tower_target_size and self._magnet_toggle == False:
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

    def is_done(self):
        return self._done

    # return observation
    def _get_obs(self):
        pass

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        out[self._arm_x, self._arm_y] = 3 - self._magnet_toggle * 1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')

    def get_current_state(self):
        return self._current_state

    def get_actions_as_dict(self):
        return {_: getattr(self.ACTIONS, _) for _ in self.ACTIONS._fields}

    def write_env_spec(self, file):
        f = open(file, 'a')
        f.write("Environment specifications:" + '\n')
        f.write(" size_x : {}".format(self._size_x) + '\n')
        f.write(" size_y : {}".format(self._size_y) + '\n')
        f.write(" cubes_cnt : {}".format(self._cubes_cnt) + '\n')
        f.write(" episode_max_length : {}".format(self._episode_max_length) + '\n')
        f.write(" finish_reward : {}".format(self._finish_reward) + '\n')
        f.write(" action_minus_reward : {}".format(self._action_minus_reward) + '\n')
        f.write(" tower_target_size : {}".format(self._tower_target_size) + '\n')
        f.write('\n')
        f.close()
