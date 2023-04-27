import os
import numpy as np
import tensorflow as tf
import time
import unicodedata

from typing import List
from enum import Enum, auto
from IPython.display import clear_output

from tf_agents.environments import py_environment
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts

ROWS = 10
COLUMNS = 10
MAX_STEPS = ROWS * COLUMNS


SHIPS = {
    2: 1,
    3: 2,
    4: 1,
    5: 1
}

INPUT_SHAPE_FLAT = (ROWS*COLUMNS*2)
MAX_HIT_COUNT = sum([x*y for x, y in SHIPS.items()])


class State(Enum):
    UNK = 0
    MISS = 1
    HIT = 2
    
    def __str__(self):
        return (
        '❔'		if self.value == self.UNK.value else
        '💧'	 if self.value == self.MISS.value else
        '🔥'	 if self.value == self.HIT.value else
        ' ')

class HiddenState(Enum):
    WATER = 0
    SHIP = 1

    def __str__(self):
        return (
        '💧'	 if self.value == self.WATER.value else
        '🚢'	if self.value == self.SHIP.value else
        ' ')

class Orientation(Enum):
    VERTICAL = auto()
    HORIZONTAL = auto()

# make static 

orientation_as_list = [x.value for x in Orientation]

class Ship:
    def __init__(self, size: int) -> None:
        self.size = size
        self.is_placed = False
        self.HP = size
        self.hit_locations = [[None, None]] * size
    
    def get_location(self):
        return self.x, self.y

    def get_size(self):
        return self.size

    def place(self, state, alignment, x, y) -> None:
        self.alignment = alignment
        self.x = x
        self.y = y
        
        if self.alignment == Orientation.HORIZONTAL.value:
            state[x, y: y+self.size] = np.full(self.size, HiddenState.SHIP.value, dtype=np.int32)
        else:
            state[y: y+self.size, x] = np.full(self.size, HiddenState.SHIP.value, dtype=np.int32)

        self.is_placed = True

    def is_placed(self) -> bool:
        return self.is_placed

    def is_already_hit(self, position) -> bool:
        return position.tolist() in self.hit_locations

    def is_hit(self, position) -> bool:
        x, y = position

        if self.alignment == Orientation.HORIZONTAL.value:
            if self.x == x and self.y <= y and y <= self.y + self.size-1:
                #self.hit_locations[self.size-self.HP] = position.tolist()
                self.HP -= 1
                return True
        else:
            if self.x == y and self.y <= x and x <= self.y + self.size-1:
                #self.hit_locations[self.size-self.HP] = position.tolist()
                self.HP -= 1
                return True
        return False
    
    def is_alive(self, x, y) -> bool:
        self.is_hit((x, y))
        return self.HP != 0

class BattleshipEnvironment(py_environment.PyEnvironment):
    
    def __init__(self):
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=ROWS*COLUMNS-1, name='action')
        
        self._observation_spec = {
            'observation': BoundedArraySpec(shape=(INPUT_SHAPE_FLAT,), dtype=np.int32, minimum=0, maximum=1),
            'valid_actions': ArraySpec(shape=(ROWS*COLUMNS, ), dtype=np.bool_, name="valid_actions")
        }

        self._state = 0
        self._episode_ended = False
        self._step_counter = tf.Variable(0)
        self._ships: List[Ship] = []
        self._hits = []
        self._misses = []
        
        self._all_actions = np.array([x for x in range(ROWS*COLUMNS)], np.int32)

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def render(self, delay=.5):
        def clear():
            os.system('cls' if os.name=='nt' else 'clear')
            for _ in range(10):
                clear_output(wait=True)
        
        def width(string):
            return sum(1+(unicodedata.east_asian_width(c) in "WF")
                for c in string)

        time.sleep(delay)
        clear()

        for state, hidden in zip(self._state, self._hidden_state):
            print(''.join(f"{str(State.HIT) if x == State.HIT.value else str(HiddenState.WATER) if x == State.MISS.value else str(State.UNK):<3}" for x in state), end="| ")
            print(''.join(f"{str(HiddenState.SHIP) if x != HiddenState.WATER.value else str(HiddenState.WATER):<3}" for x in hidden))
    
    def create_ships(self):
        self._ships.clear()
        for length, count in SHIPS.items():
            for _ in range(count):
                self._ships.append(Ship(length))
                       
    def place_ships(self):
        for ship in self._ships:
            is_occupied = True
            x = 0
            y = 0
            rdm_orien = None
            
            while is_occupied:
                rdm_orien = np.random.choice(orientation_as_list)

                if rdm_orien == Orientation.HORIZONTAL.value:
                    x = np.random.randint(ROWS)
                    y = np.random.randint(ROWS - ship.get_size() + 1)
                    val = self._hidden_state[x, y: y + ship.get_size()]
                else:
                    x = np.random.randint(COLUMNS)
                    y = np.random.randint(COLUMNS - ship.get_size() + 1)
                    val = self._hidden_state[y: y + ship.get_size(), x]
                
                is_occupied = np.any(val != HiddenState.WATER.value)
                
            
            ship.place(self._hidden_state, rdm_orien, x, y)

    def unpack_action(self, action):
        return action // ROWS, action % COLUMNS

    def get_valid_actions(self):
        return np.isin(self._all_actions, self._possible_actions)
    
    def update_valid_actions(self, action):
        self._possible_actions = np.delete(self._possible_actions, np.where(self._possible_actions == action))
    
    def get_info(self):
        return {
            'legal actions': self._possible_actions if self._possible_actions else [],
        }

    def get_state(self):
        return {
            'observation': self._observation, 
            'valid_actions': self.get_valid_actions()
        }
        
    def reset_valid_actions(self):
            """Valid actions are resetted.
            """
            self._possible_actions = self._all_actions.copy()
    
    def _reset(self):
        self._step_counter.assign(0)
        self._episode_ended = False

        self._observation = np.full(shape=(INPUT_SHAPE_FLAT,), fill_value=0, dtype=np.int32)

        self._hidden_state = np.full(shape=(ROWS, COLUMNS), fill_value=HiddenState.WATER.value, dtype=np.int32)
        self._state = np.full(shape=(ROWS, COLUMNS), fill_value=State.UNK.value, dtype=np.int32)
        self._hits = np.full(shape=(ROWS, COLUMNS), fill_value=0, dtype=np.int32)
        self._misses = np.full(shape=(ROWS, COLUMNS), fill_value=0, dtype=np.int32)
        
        self.reset_valid_actions()
        self.create_ships()
        self.place_ships()
        
        return ts.restart(self.get_state())

    def _step(self, action):  
        if self._episode_ended:
            return self.reset()     
        
        self._step_counter.assign_add(1)
        step = self._step_counter.numpy()
        
        self.update_valid_actions(action)
        x, y = self.unpack_action(action)
        action = np.array([x, y], np.int32)

        if self._hidden_state[x, y] == HiddenState.SHIP.value:
            self._state[x, y] = State.HIT.value
            self._hits[x, y] = 1
            reward = 1
        else:
            self._state[x, y] = State.MISS.value
            self._misses[x, y] = 1
            reward = -.15
        
        self._observation = np.hstack((self._misses.flatten(), self._hits.flatten()))

        TOTAL_HITS = np.count_nonzero(self._state == State.HIT.value)	
        
        if step >= MAX_STEPS or TOTAL_HITS == MAX_HIT_COUNT:
            self._episode_ended = True
            reward += (MAX_STEPS-step)
            return ts.termination(self.get_state(), reward)
        else:
            return ts.transition(self.get_state(), reward, discount=1)