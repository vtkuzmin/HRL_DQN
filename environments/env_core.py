import gym


class CoreEnv(gym.Env):
    def is_done(self): raise NotImplementedError

    def get_current_state(self): raise NotImplementedError

    def get_actions_as_dict(self): raise NotImplementedError

    def _step(self, action): raise NotImplementedError

    def _reset(self): raise NotImplementedError

    def _render(self, mode='human', close=False): raise NotImplementedError

    # def _seed(self, seed=None): raise NotImplementedError
