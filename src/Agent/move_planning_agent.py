import sys
sys.path.append("src")
from Environment.environment import Environment
from Agent.base_agent import Agent

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MovePlanningAgent(Agent):
    def __init__(self, environment):
        super().__init__()
        self.env = environment
        self.has_gold = False
        self.state = nx.DiGraph()
        self.escape_actions = None

    def reset(self):
        self.state.clear()
        percepts = self.env.reset()
        self.add_new_cell_to_state(percepts["current_location"][0])
        self.total_reward = 0
        self.has_gold = False
        self.escape_actions = None
        return percepts

    def action(self, percepts: dict) -> int:
        if percepts["glitter"]==True:
            self.has_gold = True
            print("Executing execute escape plan")
            escape_plan = self.construct_escape_plan(percepts["current_location"])
            self.escape_actions = self.convert_escape_plan_to_actions(escape_plan)
            return 4 #grab
        if percepts["current_location"][0]==[0,0]:
            if self.has_gold==True:
                return 5 #climb
        if self.has_gold == True:
            action = self.escape_actions[0]
            del self.escape_actions[0]
            return action
        else:
            actions = [0,1,2,3]
            return np.random.choice(actions)
    
    def construct_escape_plan(self, source_location):

        def _remove_superfluous_nodes(shortest_path):
            nodes = [eval(node) for node in reversed(shortest_path)]
            n=0
            while nodes[n+1][0] == [0,0]:
                del nodes[n]
                n+=1

            return [str(node) for node in reversed(nodes)]

        shortest_path = nx.astar_path(self.state, 
                                      source=str(source_location), 
                                      target=str([[0,0], 'East']),
                                      heuristic=self.manhattan_distance)
        shortest_path = _remove_superfluous_nodes(shortest_path)
        attrs = {node:"shortest_path" for node in shortest_path}
        nx.set_node_attributes(self.state, attrs, name="in_shortest_path")
        return shortest_path
    
    def run(self):
        percepts = self.reset()
        while not percepts["terminated"] == True:
            previous_location = percepts["current_location"]
            action = self.action(percepts)
            percepts = self.env.step(action=action)
            current_location = percepts["current_location"]
            self.total_reward+=percepts["reward"]
            self.update_state(previous_location, current_location)

    def update_state(self, previous_location, current_location):
        self.add_new_cell_to_state(str(current_location[0]))
        self.state.add_edge(str(previous_location), 
                    str(current_location))
        #Update state to return to previous location
        refernce_current_location = current_location.copy()
        refernce_previous_location = previous_location.copy()
        if not refernce_current_location == refernce_previous_location:
            if refernce_current_location[1] == "West":
                refernce_current_location[1] = "East"
                refernce_previous_location[1] = "East"
                self.state.add_edge(str(refernce_current_location), 
                        str(refernce_previous_location))
            elif refernce_current_location[1] == "East":
                refernce_current_location[1] = "West"
                refernce_previous_location[1] = "West"
                self.state.add_edge(str(refernce_current_location), 
                        str(refernce_previous_location))
            elif refernce_current_location[1] == "North": #2,1 North
                refernce_current_location[1] = "South" #2,1 South
                refernce_previous_location[1] = "South" #1,1 South
                self.state.add_edge(str(refernce_current_location), 
                        str(refernce_previous_location))
            elif refernce_current_location[1] == "South":
                refernce_current_location[1] = "North"
                refernce_previous_location[1] = "North"
                self.state.add_edge(str(refernce_current_location), 
                        str(refernce_previous_location))
            
    @staticmethod
    def convert_escape_plan_to_actions(escape_plan):
        orientations = [eval(node)[1] for node in escape_plan]
        escape_actions = []
        for n,orientation in enumerate(orientations[:-1]):
            if orientations[n+1] == orientation:
                escape_actions.append(0)
            elif orientation == "North":
                if orientations[n+1] == "West":
                    escape_actions.append(1)
                elif orientations[n+1] == "East":
                    escape_actions.append(2)
                else:
                    raise ValueError(f"Cannot move directly from {orientation} to {orientations[n+1]}")
            elif orientation == "South":
                if orientations[n+1] == "West":
                    escape_actions.append(2)
                elif orientations[n+1] == "East":
                    escape_actions.append(1)
                else:
                    raise ValueError(f"Cannot move directly from {orientation} to {orientations[n+1]}")
            elif orientation == "West":
                if orientations[n+1] == "North":
                    escape_actions.append(2)
                elif orientations[n+1] == "South":
                    escape_actions.append(1)
                else:
                    raise ValueError(f"Cannot move directly from {orientation} to {orientations[n+1]}")
            elif orientation == "East":
                if orientations[n+1] == "North":
                    escape_actions.append(1)
                elif orientations[n+1] == "South":
                    escape_actions.append(2)
                else:
                    raise ValueError(f"Cannot move directly from {orientation} to {orientations[n+1]}")
        return escape_actions


    def add_new_cell_to_state(self, cell):
        if not self.state.has_node(f"[{cell}, East]"):
            edges = []
            for h in ["East", "West"]:
                for v in ["North", "South"]:
                    edges.extend([(f"[{cell}, '{h}']",f"[{cell}, '{v}']"), 
                                (f"[{cell}, '{v}']", f"[{cell}, '{h}']")])
            self.state.add_edges_from(edges)

    @staticmethod
    def manhattan_distance(source, target):
        return abs(eval(source)[0][0] - eval(target)[0][0]) + abs(eval(source)[0][1] - eval(target)[0][1])

if __name__=="__main__":
    env_object = Environment(width=4, 
                height=4, 
                allowClimbWithoutGold=True, 
                pitProb=0.2)
    agent = MovePlanningAgent(environment=env_object)
    agent.run()
    print(agent.total_reward)


