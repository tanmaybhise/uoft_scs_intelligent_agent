import sys
sys.path.append("src")
from Environment.environment import Environment
from Agent.base_agent import Agent

import numpy as np

class NaiveAgent(Agent):
    def __init__(self, environment):
        super().__init__()
        self.env = environment

    def action(self, percepts: dict) -> int:
        actions = list(self.env.get_action_set().keys())
        return np.random.choice(actions)
    
    def run(self):
        percepts = self.env.reset()
        self.total_reward = 0
        while not percepts["terminated"] == True:        
            action = self.action(percepts)
            percepts = self.env.step(action=action)
            self.total_reward+=percepts["reward"]

if __name__=="__main__":
    env_object = Environment(width=4, 
                 height=4, 
                 allowClimbWithoutGold=False, 
                 pitProb=0.2)
    agent = NaiveAgent(environment=env_object)
    agent.run()
    print(agent.total_reward)