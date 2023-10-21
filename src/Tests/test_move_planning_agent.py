import sys
sys.path.append("..")
from Environment.environment import Environment
from Agent.move_planning_agent import MovePlanningAgent
from config import Config

import numpy as np
import networkx as nx

def test_move_planning_agent():
    agent = _get_agent_objevt()
    #Get to the gold
    total_reward, percepts = _perform_agent_step(agent, action=0)
    total_reward, percepts = _perform_agent_step(agent, action=1)
    total_reward, percepts = _perform_agent_step(agent, action=0)
    total_reward, percepts = _perform_agent_step(agent, action=0)
    #Execute escape plan
    action = agent.action(percepts)
    assert action==4
    for _ in range(7):
        total_reward, percepts = _perform_agent_step(agent, action=action)
        action = agent.action(percepts)
    assert action == 5
    total_reward, percepts = _perform_agent_step(agent, action=action)
    assert percepts["reward"] == 999
    assert total_reward == 988

def _perform_agent_step(agent, action):
    previous_location = [agent.env.get_locations()["agent"][0], agent.env.get_agent_orientation()]
    percepts = agent.env.step(action=action)
    current_location = percepts["current_location"]
    agent.total_reward+=percepts["reward"]
    agent.update_state(previous_location, current_location)
    return agent.total_reward, percepts

def _get_agent_objevt():    
    env_object = Environment(width=4, 
                    height=4, 
                    allowClimbWithoutGold=True, 
                    pitProb=0.2,
                    debug=True)
    agent = MovePlanningAgent(environment=env_object)
    agent.env.set_locations({'agent': [[0, 0]],
                            'pit': [[3, 3]],
                            'gold': [[2, 1]],
                            'wumpus': [[3, 1]]})
    agent.add_new_cell_to_state(agent.env.get_locations()["agent"][0])
    return agent
