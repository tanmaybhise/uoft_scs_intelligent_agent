import numpy as np
from config import Config

class Environment():
    def __init__(self, width: 4, 
                 height: 4, 
                 allowClimbWithoutGold: True, 
                 pitProb: 0.2,
                 debug=False):
        self.width = width
        self.height = height
        self.allowClimbWithoutGold = allowClimbWithoutGold
        self.pitProb = pitProb
        self.debug = debug
        self.actions = self.get_action_set()
        self.reset()

    def reset(self):
        self.__agent_orientation = Config.agent_default_orientation
        self.agent_arrows_left = 1
        self.agent_has_the_gold = False
        self.wumpus_killed = False
        self.element_codes = self.__generate_element_codes()
        self.__state, self.__locations = self.__initialize_locations()
        percepts = self.update_percepts()
        return percepts

    def step(self, action):
        assert len(self.__locations["agent"]) == 1, NotImplementedError("There must only be one agent")
        assert len(self.__locations["gold"]) <= 1, NotImplementedError("There must only be one gold")
        assert len(self.__locations["wumpus"]) <= 1, NotImplementedError("There must only be one wumpus")
        agent_location = self.__locations["agent"][0].copy()
        wumpus_location = self.__locations["wumpus"][0].copy()

        percepts = {"stench":False, 
                    "breeze":False, 
                    "glitter":False, 
                    "bump":False, 
                    "scream":False,
                    "current_location": [[],None],
                    "reward": -1,
                    "terminated": False}
        
        if self.actions[action] == "shoot":
            if self.agent_arrows_left > 0:
                self.agent_arrows_left-=1
                if self.__agent_orientation == "East":
                    if agent_location[0] == wumpus_location[0]:
                        if agent_location[1] < wumpus_location[1]:
                            percepts["scream"] = True
                            self.wumpus_killed = True
                            percepts["reward"]+=-10
                elif self.__agent_orientation == "West":
                    if agent_location[0] == wumpus_location[0]:
                        if agent_location[1] > wumpus_location[1]:
                            percepts["scream"] = True
                            self.wumpus_killed = True
                            percepts["reward"]+=-10
                elif self.__agent_orientation == "North":
                    if agent_location[1] == wumpus_location[1]:
                        if agent_location[0] < wumpus_location[0]:
                            percepts["scream"] = True
                            self.wumpus_killed = True
                            percepts["reward"]+=-10
                elif self.__agent_orientation == "South":
                    if agent_location[1] == wumpus_location[1]:
                        if agent_location[0] > wumpus_location[0]:
                            percepts["scream"] = True
                            self.wumpus_killed = True
                            percepts["reward"]+=-10

        elif self.actions[action] == "turn_left":
            if self.__agent_orientation == "East":
                self.__agent_orientation = "North"
            elif self.__agent_orientation == "West":
                self.__agent_orientation = "South"
            elif self.__agent_orientation == "North":
                self.__agent_orientation = "West"
            elif self.__agent_orientation == "South":
                self.__agent_orientation = "East"

        elif self.actions[action] == "turn_right":
            if self.__agent_orientation == "East":
                self.__agent_orientation = "South"
            elif self.__agent_orientation == "West":
                self.__agent_orientation = "North"
            elif self.__agent_orientation == "North":
                self.__agent_orientation = "East"
            elif self.__agent_orientation == "South":
                self.__agent_orientation = "West"

        elif self.actions[action] == "forward":
            if self.__agent_orientation == "East":
                agent_location[1]+=1
            elif self.__agent_orientation == "West":
                agent_location[1]-=1
            elif self.__agent_orientation == "North":
                agent_location[0]+=1
            elif self.__agent_orientation == "South":
                agent_location[0]-=1

            if ((agent_location[1] < self.width) &\
                (agent_location[1] >= 0) &\
                (agent_location[0] < self.height) &\
                (agent_location[0] >= 0)):
                self.__locations["agent"][0] = agent_location
            else:
                percepts["bump"] = True

        elif self.actions[action] == "grab":
            if not self.agent_has_the_gold:
                gold_location = self.__locations["gold"][0]
                if agent_location in self.__locations["gold"]:
                    self.__locations["gold"].remove(gold_location)
                    self.agent_has_the_gold = True

        elif self.actions[action] == "climb":
            if agent_location==[0,0]:
                if self.agent_has_the_gold:
                    percepts["reward"]+=1000
                    percepts["terminated"] = True
                else:
                    if self.allowClimbWithoutGold:
                        percepts["terminated"] = True

        else:
            raise NotImplementedError(f"action `{action}` is not recognized.\
                                      Allowed options {self.actions}")
        
        percepts = self.update_percepts(percepts=percepts)
        
        self.__state = self.__update_state(self.__locations)

        return percepts
    
    def update_percepts(self, percepts={"stench":False, 
                                        "breeze":False, 
                                        "glitter":False, 
                                        "bump":False, 
                                        "scream":False,
                                        "current_location": [[],None],
                                        "reward": 0,
                                        "terminated": False}):
        agent_location = self.__locations["agent"][0].copy()
        wumpus_location = self.__locations["wumpus"][0].copy()
        if agent_location in self.__locations["pit"]:
            percepts["reward"] += -1000
            percepts["terminated"] = True

        if agent_location in self.__locations["wumpus"]:
            if not self.wumpus_killed:
                percepts["reward"] += -1000
                percepts["terminated"] = True

        if not self.agent_has_the_gold:
            if agent_location in [
                                self.__locations["gold"][0]
                                ]:
                percepts["glitter"] = True
        
        if agent_location in [
                              wumpus_location, 
                              [wumpus_location[0]+1, wumpus_location[1]],
                              [wumpus_location[0]-1, wumpus_location[1]],
                              [wumpus_location[0], wumpus_location[1]+1],
                              [wumpus_location[0], wumpus_location[1]-1],

                              ]:
            percepts["stench"] = True

        for pit_loc in self.__locations["pit"]:
            if agent_location in [
                            [pit_loc[0]+1, pit_loc[1]],
                            [pit_loc[0]-1, pit_loc[1]],
                            [pit_loc[0], pit_loc[1]+1],
                            [pit_loc[0], pit_loc[1]-1],
                            ]:
                percepts["breeze"] = True
        
        percepts["current_location"][0] = agent_location
        percepts["current_location"][1] = self.__agent_orientation

        return percepts
         
    def set_locations(self, locations):
        if self.debug:
            self.__locations = locations
            self.__state = self.__update_state(self.__locations)
        else:
            raise ValueError("set_state method if not allowed when debug=False. \
                             Set debug=True to use this method.")
        
    def get_locations(self):
        if self.debug:
            return self.__locations
        else:
            raise ValueError("get_locations method if not allowed when debug=False. \
                             Set debug=True to use this method.")
    
    def get_agent_orientation(self):
        if self.debug:
            return self.__agent_orientation
        else:
            raise ValueError("get_agent_orientation method if not allowed when debug=False. \
                             Set debug=True to use this method.")
    
    @staticmethod
    def get_action_set():
        return {
            0: "forward", 
            1: "turn_left", 
            2: "turn_right", 
            3: "shoot", 
            4: "grab",
            5: "climb"
            }
    
    def __generate_element_codes(self):
        if self.__agent_orientation == "East":
            agent_direction = "ᐅ"
        elif self.__agent_orientation == "West":
            agent_direction = "ᐊ"
        elif self.__agent_orientation == "North":
            agent_direction = "ᐃ"
        elif self.__agent_orientation == "South":
            agent_direction = "ᐁ"
        else:
            raise ValueError(f"agent_orientation {self.__agent_orientation} is not recognized")

        element_codes = {
                        "agent": f"{agent_direction}",
                        "wumpus": f"W{[':(' if self.wumpus_killed else ''][0]}",
                        "pit":"P",
                        "gold": "G"
                        }
        return element_codes
    
    def __generate_pit_locations(self):
        h,w = np.meshgrid(np.arange(1,self.height), np.arange(1,self.width))
        pit_location = []
        for pit_loc in list(zip(h.flatten(), w.flatten())):
            if np.random.random() < self.pitProb:
                pit_location.append(list(pit_loc))
        return pit_location
    
    def __random_location_generator(self):
        locations = {"agent": [[0,0]],
            "pit": self.__generate_pit_locations(),
            "gold": [[np.random.choice(np.arange(1,self.height), 
                            p=1/(self.height-1)*np.ones(self.height-1)),
                     np.random.choice(np.arange(1,self.width), 
                            p=1/(self.width-1)*np.ones(self.width-1))]],
            "wumpus": [[np.random.choice(np.arange(1,self.height), 
                            p=1/(self.height-1)*np.ones(self.height-1)),
                      np.random.choice(np.arange(1,self.width),
                            p=1/(self.width-1)*np.ones(self.width-1))]]}
        return locations
    
    def __initialize_locations(self):
        locations = self.__random_location_generator()
        state = self.__update_state(locations)
        return state, locations
    
    def __update_state(self, locations):
        self.element_codes = self.__generate_element_codes()
        state = np.zeros(self.height*self.width, dtype=object).\
                    reshape(self.height,self.width)
        state[:,:] = "-"   
        for element,location in locations.items():
            for loc in location:
                if state[loc[0], loc[1]] != "-":
                    existing_element = state[loc[0], loc[1]]
                    state[loc[0], loc[1]] = existing_element+\
                                                        self.element_codes[element]
                else:
                    state[loc[0], loc[1]] = self.element_codes[element]
        return state

    def __repr__(self) -> str:
        viz_state = np.flipud(self.__state)
        return "\n".join(map(str, viz_state.tolist()))