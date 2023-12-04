import sys
sys.path.append("src")
from Environment.environment import Environment
from Agent.base_agent import Agent

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import torch
from pomegranate.bayesian_network import BayesianNetwork
import os

import warnings
warnings.filterwarnings("ignore")

class ProbAgent(Agent):
    def __init__(self, environment):
        super().__init__()
        self.env = environment
        self.has_gold = False
        self.state = nx.DiGraph()
        self.escape_actions = None
        self.heard_scream = False
        self.init_memory()

    def init_memory(self):
        self.wumpus_model = None
        self.pit_model = None
        self.stench_locations = {}
        self.breeze_locations = {}
        self.observed_pit_locations = {}
        self.observed_wumpus_locations = {}
        self.grid_size = self.env.width
        self.n = 0

        cells = []
        for c1 in range(0,self.grid_size):
            for c2 in range(0,self.grid_size):
                cells.append([c1,c2])

        breeze_columns = [f"breeze_{cell}" for cell in cells]
        self.pit_columns = [f"pit_{cell}" for cell in cells]

        stench_columns = [f"stench_{cell}" for cell in cells]
        self.wumpus_columns = [f"wumpus_{cell}" for cell in cells]
    
        if os.path.isfile("src\Agent\memory\prob_agent_wumpus_memory.csv"):
            self.wumpus_memory = pd.read_csv("src\Agent\memory\prob_agent_wumpus_memory.csv", index_col=[0])
            self.n = max(self.wumpus_memory.index) + 1
        else:
            self.wumpus_memory = pd.DataFrame(columns=stench_columns+self.wumpus_columns)
            self.wumpus_memory.index.name = "episode"
            self.wumpus_memory.loc[-1] = 0 #prior
            self.wumpus_memory.loc[-2] = 1 #prior
        
        if os.path.isfile("src\Agent\memory\prob_agent_pit_memory.csv"):
            self.pit_memory = pd.read_csv("src\Agent\memory\prob_agent_pit_memory.csv", index_col=[0])
        else:
            self.pit_memory = pd.DataFrame(columns=breeze_columns+self.pit_columns)
            self.pit_memory.index.name = "episode"
            self.pit_memory.loc[-1] = 0 #prior
            self.pit_memory.loc[-2] = 1 #prior

        if os.path.isfile(r"src\Agent\memory\reward_history.csv"):
            self.reward_history = pd.read_csv(r"src\Agent\memory\reward_history.csv", index_col=[0])
        else:
            self.reward_history = pd.DataFrame(columns=['total_reward'])
            self.reward_history.index.name = "episode"
        
        #self.update_belief()

    def update_belief(self):
        structure = [()]*(self.grid_size*self.grid_size) \
            + [tuple([i for i in range(self.grid_size*self.grid_size)])] \
                * self.grid_size*self.grid_size
        
        #Remove points where pit was not seen
        faulty_data = list(self.pit_memory.index[(self.pit_memory[self.pit_memory.columns[self.pit_memory.columns.str.contains("pit")]] == 0).sum(axis=1) == 16])
        faulty_data = [inx for inx in faulty_data if inx not in [-1,-2]]
        pit_memory = self.pit_memory.drop(faulty_data)
        pit_states = pit_memory.to_numpy().astype(int)

        self.pit_model = BayesianNetwork(structure=structure)
        self.pit_model.fit(pit_states)

        #Remove points where wumpus was not seen
        faulty_data = list(self.wumpus_memory.index[(self.wumpus_memory[self.wumpus_memory.columns[self.wumpus_memory.columns.str.contains("wumpus")]] == 0).sum(axis=1) == 16])
        faulty_data = [inx for inx in faulty_data if inx not in [-1,-2]]
        wumpus_memory = self.wumpus_memory.drop(faulty_data)
        wumpus_states = wumpus_memory.to_numpy().astype(int)

        self.wumpus_model = BayesianNetwork(structure=structure)
        self.wumpus_model.fit(wumpus_states)

    def predict_wumpus(self):
        if self.wumpus_model is not None:
            wumpus_states = self.wumpus_memory.loc[[self.n]].to_numpy().astype(int)
            mask = np.array([True]*(2*self.grid_size*self.grid_size)).reshape(1,2*self.grid_size*self.grid_size)
            mask[:,self.grid_size*self.grid_size:] = False
            masked_wumpus_states = torch.masked.MaskedTensor(torch.tensor(wumpus_states), mask=torch.tensor(mask))
            preds = self.wumpus_model.predict(masked_wumpus_states)
            possible_wumpus_location = pd.DataFrame(preds[:,self.grid_size*self.grid_size:], columns=self.wumpus_columns, index=[self.n])
            possible_wumpus_location = possible_wumpus_location.T.loc[(possible_wumpus_location.T == 1).values.flatten(), :].index
            return [eval(loc.split("_")[1]) for loc in possible_wumpus_location]
        else:
            return []
    
    def predict_pit(self):
        if self.pit_model is not None:
            pit_states = self.pit_memory.loc[[self.n]].to_numpy().astype(int)
            mask = np.array([True]*(2*self.grid_size*self.grid_size)).reshape(1,2*self.grid_size*self.grid_size)
            mask[:,self.grid_size*self.grid_size:] = False
            masked_pit_states = torch.masked.MaskedTensor(torch.tensor(pit_states), mask=torch.tensor(mask))
            preds = self.pit_model.predict(masked_pit_states)
            possible_pit_location = pd.DataFrame(preds[:,self.grid_size*self.grid_size:], columns=self.pit_columns, index=[self.n])
            possible_pit_location = possible_pit_location.T.loc[(possible_pit_location.T == 1).values.flatten(), :].index
            return [eval(loc.split("_")[1]) for loc in possible_pit_location]
        else:
            return []

    def reset(self):
        self.state.clear()
        percepts = self.env.reset()
        self.add_new_cell_to_state(percepts["current_location"][0])
        self.total_reward = 0
        self.has_gold = False
        self.escape_actions = None
        self.heard_scream = False
        self.executing_escape_plan = False
        return percepts

    def action(self, percepts: dict) -> int:
        if percepts["glitter"]==True:
            self.has_gold = True
            #print("Executing execute escape plan")
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
            if not self.executing_escape_plan:
                possible_pit_location = self.predict_pit()
                possible_wumpus_location = self.predict_wumpus()
                next_location_on_forward = self.get_next_location_on_forward(percepts["current_location"])
                possible_locations = self.get_all_possible_locations(percepts["current_location"])
                visites_cells = list(set([node[1:7] for node in list(dict(self.state.nodes).keys())]))
                visites_cells = [eval(cell) for cell in visites_cells]
                
                #remove wumpus locations from possible locations
                if not self.heard_scream:
                    for loc in possible_wumpus_location:
                        try:
                            possible_locations.remove(loc)
                        except ValueError:
                            pass

                #remove pit locations from possible locations
                for loc in possible_pit_location:
                    try:
                        possible_locations.remove(loc)
                    except ValueError:
                        pass

                #remove already visited locations from possible locations
                for loc in visites_cells:
                    try:
                        possible_locations.remove(loc)
                    except ValueError:
                        pass

                if possible_locations != []:
                    action_choices = [0,1,2,3] 
                    #avoid blind shooting
                    if ((self.heard_scream) \
                        | ~(percepts["current_location"][0] in possible_wumpus_location)):
                        action_choices.remove(3)
                    
                    #avoid falling into wumpus
                    if ((next_location_on_forward in possible_wumpus_location) \
                        & ~(self.heard_scream)):
                        action_choices.remove(0)

                    #avoid falling into pit
                    if (next_location_on_forward in possible_pit_location):
                        action_choices.remove(0)

                    random_action = np.random.choice(action_choices)
                    return random_action
                else:
                    #print("Executing execute escape plan")
                    escape_plan = self.construct_escape_plan(percepts["current_location"])
                    self.escape_actions = self.convert_escape_plan_to_actions(escape_plan)
                    return 4 #grab
            else:
                if percepts["current_location"][0]==[0,0]:
                    return 5 #climb
                action = self.escape_actions[0]
                del self.escape_actions[0]
                return action
                        
    def get_next_location_on_forward(self, current_location):
        current_location = [[0, 3], 'South']
        next_location_on_forward = current_location[0].copy()
        reference_x,reference_y  = current_location[0][0], current_location[0][1]
        if current_location[1] == "North":
            if not reference_x == self.grid_size-1:
                next_location_on_forward = [reference_x+1, reference_y]
        if current_location[1] == "South":
            if not reference_x == 0:
                next_location_on_forward = [reference_x-1, reference_y]
        if current_location[1] == "East":
            if not reference_y == self.grid_size-1:
                next_location_on_forward = [reference_x, reference_y+1]
        if current_location[1] == "West":
            if not reference_y == 0:
                next_location_on_forward = [reference_x, reference_y-1]
        return next_location_on_forward
    
    def get_all_possible_locations(self, current_location):
        N = [(min(self.grid_size-1, current_location[0][0]+1)), current_location[0][1]]
        S = [(max(0, current_location[0][0]-1)), current_location[0][1]]
        E = [current_location[0][0], (min(self.grid_size-1, current_location[0][1]+1))]
        W = [current_location[0][0], max(0, current_location[0][1]-1)]
        possible_locations = [N,S,E,W]
        return possible_locations
    
    def construct_escape_plan(self, source_location):
        self.executing_escape_plan = True

        def _remove_superfluous_nodes(shortest_path):
            nodes = [eval(node) for node in reversed(shortest_path)]

            for n in range(len(nodes)-1):
                try:
                    if nodes[n+1][0] == [0,0]:
                        del nodes[n]
                    else:
                        break
                except IndexError:
                    break

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
        self.stench_locations[self.n] = []
        self.breeze_locations[self.n] = []
        self.observed_wumpus_locations[self.n] = []
        self.observed_pit_locations[self.n] = []
        self.wumpus_memory.loc[self.n] = 0
        self.pit_memory.loc[self.n] = 0
        percepts = self.reset()
        while not percepts["terminated"][0] == True:
            previous_location = percepts["current_location"]
            action = self.action(percepts)
            percepts = self.env.step(action=action)
            if percepts["scream"]:
                self.heard_scream = True

            if percepts["stench"]:
                if self.n in self.stench_locations:
                    self.stench_locations[self.n].append(f"""stench_{percepts["current_location"][0]}""")
                else:
                    self.stench_locations[self.n] = [f"""stench_{percepts["current_location"][0]}"""]

            if percepts["breeze"]:
                if self.n in self.breeze_locations:
                    self.breeze_locations[self.n].append(f"""breeze_{percepts["current_location"][0]}""")
                else:
                    self.breeze_locations[self.n] = [f"""breeze_{percepts["current_location"][0]}"""]

            if percepts["terminated"][0] == True:
                if percepts["terminated"][1] == "wumpus":
                    if self.n in self.observed_wumpus_locations:
                        self.observed_wumpus_locations[self.n].append(f"""wumpus_{percepts["current_location"][0]}""")
                    else:
                        self.observed_wumpus_locations[self.n] = [f"""wumpus_{percepts["current_location"][0]}"""]
                elif percepts["terminated"][1] == "pit":
                    if self.n in self.observed_pit_locations:
                        self.observed_pit_locations[self.n].append(f"""pit_{percepts["current_location"][0]}""")
                    else:
                        self.observed_pit_locations[self.n] = [f"""pit_{percepts["current_location"][0]}"""]

            current_location = percepts["current_location"]
            self.total_reward+=percepts["reward"]
            self.update_state(previous_location, current_location)
            self.update_memory()
        
        self.n = max(self.pit_memory.index)+1
        if self.n > 900:
            self.update_belief()
        
        self.pit_memory.to_csv("src\Agent\memory\prob_agent_pit_memory.csv")
        self.wumpus_memory.to_csv("src\Agent\memory\prob_agent_wumpus_memory.csv")
        self.reward_history.to_csv(r"src\Agent\memory\reward_history.csv")

    def update_memory(self):
        self.stench_locations[self.n] = list(set([str(loc) for loc in self.stench_locations[self.n]]))
        self.breeze_locations[self.n] = list(set([str(loc) for loc in self.breeze_locations[self.n]]))
        self.reward_history.loc[self.n] = [self.total_reward]
        self.wumpus_memory.loc[self.n, self.stench_locations[self.n] + self.observed_wumpus_locations[self.n]] = 1
        self.wumpus_memory = self.wumpus_memory.fillna(0)
        self.pit_memory.loc[self.n, self.breeze_locations[self.n] + self.observed_pit_locations[self.n]] = 1
        self.pit_memory = self.pit_memory.fillna(0)

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
            elif refernce_current_location[1] == "North":
                refernce_current_location[1] = "South"
                refernce_previous_location[1] = "South"
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
    from matplotlib import pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    def main():
        env_object = Environment(width=4, 
                    height=4, 
                    allowClimbWithoutGold=True, 
                    pitProb=0.2)
        agent = ProbAgent(environment=env_object)
        n_episodes = 1
        for _ in tqdm(range(n_episodes), desc=f"Running agent {n_episodes} episodes"):
            agent.run()
        sns.lineplot(x=agent.reward_history.index, y=agent.reward_history.total_reward)
        plt.title("Reward History")
        plt.show()

    main()
