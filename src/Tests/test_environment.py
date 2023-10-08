import sys
sys.path.append("..")
from Environment.environment import Environment
from Agent.naive_agent import NaiveAgent

import numpy as np

def test_forward():
    env = _get_environment_object()
    env.step(0)
    new_locations = env.get_locations()
    assert new_locations == {'agent': [[0, 1]], 
                             'pit': [[0, 3]], 
                             'gold': [[1, 1]], 
                             'wumpus': [[3, 2]]}
    agent_at_bottom_right_corner_locations = {'agent': [[0, 3]], 
                'pit': [[0, 3]], 
                'gold': [[1, 1]], 
                'wumpus': [[3, 2]]}

    env.set_locations(locations=agent_at_bottom_right_corner_locations)
    percepts = env.step(0)
    assert percepts["bump"] == True
    new_locations = env.get_locations()
    assert new_locations == agent_at_bottom_right_corner_locations

def test_turn_left_and_turn_right():
    env = _get_environment_object()
    assert env.get_agent_orientation() == "East"
    env.step(1)
    assert env.get_agent_orientation() == "North"
    env.step(1)
    assert env.get_agent_orientation() == "West"
    env.step(1)
    assert env.get_agent_orientation() == "South"
    env.step(2)
    assert env.get_agent_orientation() == "West"
    env.step(2)
    assert env.get_agent_orientation() == "North"
    env.step(2)
    assert env.get_agent_orientation() == "East"

def test_grab():
    env = _get_environment_object()
    env.step(0)
    env.step(1)
    percepts = env.step(0)

    assert env.get_locations()["agent"] == env.get_locations()["gold"]
    assert percepts["glitter"] == True

    percepts = env.step(4)

    assert env.get_locations()["gold"] == []
    assert env.agent_has_the_gold == True

def test_shoot():
    env = _get_environment_object()
    env.step(0)
    env.step(0)
    env.step(1)
    assert env.agent_arrows_left == 1
    percepts = env.step(3)
    assert percepts["scream"] == True
    assert percepts["reward"] == -11
    assert env.agent_arrows_left == 0

def test_climb():
    env = _get_environment_object()
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(4)
    env.step(1)
    env.step(0)
    env.step(1)
    env.step(0)
    percepts = env.step(5)
    assert env.agent_has_the_gold == True
    assert percepts["terminated"] == True
    assert percepts["reward"] == 999

def test_pit():
    env = _get_environment_object()
    env.step(0)
    percepts = env.step(0)
    assert percepts["breeze"] == True
    env.step(1)
    env.step(0)
    percepts = env.step(2)
    assert percepts["breeze"] == False
    percepts = env.step(0)
    assert percepts["breeze"] == True
    env.step(2)
    percepts = env.step(0)
    assert percepts["terminated"] == True
    assert percepts["reward"] == -1001

def test_wumpus():
    env = _get_environment_object()
    env.step(0)
    percepts = env.step(0)
    assert percepts["stench"] == False
    env.step(1)
    env.step(0)
    percepts = env.step(0)
    assert percepts["stench"] == True
    percepts = env.step(0)
    assert percepts["stench"] == True
    assert percepts["reward"] == -1001
    assert percepts["terminated"] == True

def _get_environment_object():
    env = Environment(width=4, 
                 height=4, 
                 allowClimbWithoutGold=True, 
                 pitProb=0.2,
                 debug=True)

    locations = {'agent': [[0, 0]], 
                'pit': [[0, 3]], 
                'gold': [[1, 1]], 
                'wumpus': [[3, 2]]}

    env.set_locations(locations=locations)

    return env