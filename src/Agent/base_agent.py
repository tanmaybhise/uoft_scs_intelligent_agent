import abc

class Agent(metaclass=abc.ABCMeta):
    def __init__(self):
        self.total_reward=0

    @abc.abstractmethod
    def action(self, environment_state):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}\nReward: {self.total_reward}"