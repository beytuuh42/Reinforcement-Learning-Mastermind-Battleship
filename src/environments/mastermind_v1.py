import os
import time

import numpy as np
import tensorflow as tf

from enum import Enum
from typing import Union
from collections import Counter

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec, ArraySpec
from tf_agents.trajectories import time_step as ts

from IPython.display import clear_output

class Feedback(Enum):
    WRONG_COLOR_AND_POSITION = 0
    CORRECT_COLOR_WRONG_POSITION = 1
    CORRECT_COLOR_AND_POSITION = 2

    def __str__(self):
        return (
        '❌' if self.value == self.WRONG_COLOR_AND_POSITION.value else
        '⬜' if self.value == self.CORRECT_COLOR_WRONG_POSITION.value else
        '⬛' if self.value == self.CORRECT_COLOR_AND_POSITION.value else
            ' ')

class Color(Enum):
    YELLOW  = 0
    ORANGE  = 1
    RED     = 2
    BLUE    = 3
    GREEN   = 4
    BROWN   = 5

    def __str__(self):
        return (
        '🟨' if self.value == self.YELLOW.value else
        '🟧' if self.value == self.ORANGE.value else
        '🟥' if self.value == self.RED.value else
        '🟦' if self.value == self.BLUE.value else
        '🟩' if self.value == self.GREEN.value else
        '🟫' if self.value == self.BROWN.value else
            ' ')

TOTAL_COLORS_AMOUNT = len(Color)
TOTAL_COLORS_TO_GUESS = 4
TOTAL_ROUNDS = 10
INPUT_SHAPE = (2, TOTAL_COLORS_TO_GUESS)

MAX_GUESSES = TOTAL_COLORS_AMOUNT**TOTAL_COLORS_TO_GUESS


class MastermindEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=MAX_GUESSES-1, name='action')
        
        # change minimum and maximum from 4 elems to dynamic based on TOTAL_COLORS_TO_GUESS
        self._observation_spec = {
            'observation': BoundedArraySpec(shape=INPUT_SHAPE, dtype=np.int32, 
                minimum=[ [0]*TOTAL_COLORS_TO_GUESS, [0]*TOTAL_COLORS_TO_GUESS   ], 
                maximum=[	[TOTAL_COLORS_AMOUNT-1] *TOTAL_COLORS_TO_GUESS, 
                            [TOTAL_COLORS_TO_GUESS, TOTAL_COLORS_TO_GUESS] + [0]*(TOTAL_COLORS_TO_GUESS-2)
                ], name='observation'),
            'valid_actions': ArraySpec(shape=(MAX_GUESSES, ), dtype=np.bool_, name="valid_actions")
        }
        
        self._all_actions = np.array([x for x in range(MAX_GUESSES)], np.int32)
        self._state = 0
        self._episode_ended = False
        self._step_counter = tf.Variable(0)
        self._possible_actions = []
        self._complete_states = []

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def render(self, delay=.5):
        """Clear console and display game state with delay

        Args:
            delay (float, optional): Delay for each action. Defaults to .5.
        """
        def clear():
            os.system('cls' if os.name=='nt' else 'clear')
            for _ in range(10):
                clear_output(wait=True)

        time.sleep(delay)
        clear()

        print(f"{'#':^6}| ", end=" ")
        for i in range(TOTAL_COLORS_TO_GUESS):
            print(f"{i+1:>2}", end=" ")

        print(f"{'|':>3}", end="  ")
        print("Feedback")
        print('-'*40)

        for i, state in enumerate(self._complete_states):
            print(f"{i+1:^6}| ", end=" ")
            
            for guess in state:
                print(str(Color(guess)), end=" ")
            print(f"{'|':>3}", end="  ")
            
            black, white = self.calculate_score("".join([str(s) for s in state]), self._code)

            for _ in range(black):
                print(str(Feedback.CORRECT_COLOR_AND_POSITION), end=" ")
            
            for _ in range(white):
                print(str(Feedback.CORRECT_COLOR_WRONG_POSITION), end=" ")
                
            for _ in range((TOTAL_COLORS_TO_GUESS - black - white)):
                print(str(Feedback.WRONG_COLOR_AND_POSITION), end=" ")
                
            print()

        print('-'*40)
        print(f"{'Code':^6}| ", end=" ")
        
        for val in self._code:
            print(str(Color(int(val))), end=" ")
        print()

    def calculate_score(self, p, q):
        hits = sum(p_i == q_i for p_i, q_i in zip(p, q))
        misses = sum((Counter(p) & Counter(q)).values()) - hits
        return hits, misses	

    def get_action_index_as_array(self, state, as_int=False) -> Union[str, np.ndarray]:	
        """Convert action index (value between 0 and TOTAL_AMOUNT_COLORS**TOTAL_COLORS_TO_GUESS-1), into an array of color values, i.e. [1,2,3,4].
        The array can be either of string or int values.

        Args:
            state (_type_): Action to convert.
            as_int (bool, optional): Bool to decide whether the output should consist of string or int values. Defaults to False.

        Returns:
            Union[str, np.ndarray]: Converted action as string, or array.
        """
        assert(0 <= state <= MAX_GUESSES-1), "Number is out of boundary."
        digits = []
        while state > 0:
            digits.append(str(state % TOTAL_COLORS_AMOUNT))
            state = state // TOTAL_COLORS_AMOUNT
        arr = "".join(reversed(digits)).zfill(TOTAL_COLORS_TO_GUESS)
        return arr if not as_int else np.array([int(x) for x in arr], np.int32)

    def get_code(self) -> str:
        """Generates a code from a random uniform distribution, where each value is in range [0, TOTAL_COLORS_AMOUNT], as a string.
        
        Returns:
            str: A secret code.
        """
        gen_code = tf.random.uniform(minval=0, maxval=TOTAL_COLORS_AMOUNT, dtype=np.int32, shape=[4]).numpy()
        self._code = "".join([str(x) for x in gen_code])

    def get_info(self):
        return {
            'code': self._code,
            'legal actions': self._possible_actions
        }

    def get_state(self):
        return {
            'observation': self._state, 
            'valid_actions': self.get_valid_actions_mask()
        }
    
    def update_valid_actions(self, guess, feedback):
        """Updates valid actions for the current step, based on previous valid actions and current action.

        Args:
            guess (_type_): Current action.
            feedback (_type_): Score of the current action.
        """

        new_states = np.array([state for state in self._possible_actions if self.calculate_score(guess, self.get_action_index_as_array(state)) == feedback], np.int32)
        self._possible_actions = new_states

    def get_valid_actions_mask(self):
        """Get a mask for valid actions.

        Returns:
            np.ndarray, bool: A mask for valid actions.
        """
        return np.isin(self._all_actions, self._possible_actions)

    def reset_valid_actions(self):
        """Valid actions are resetted.
        """
        self._possible_actions = self._all_actions.copy()

    def _reset(self):
        self.get_code()
        self.reset_valid_actions()
        self._step_counter.assign(0)
        self._episode_ended = False
        self._round_counter = 0
        self._state = np.zeros(shape=INPUT_SHAPE, dtype=np.int32)
        self._complete_states = []

        return ts.restart(self.get_state())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        reward = 0
        reward -= 1
        
        self._step_counter.assign_add(1)
        self._round_counter += 1
        action_as_string = self.get_action_index_as_array(action)
        action_as_int = self.get_action_index_as_array(action, True)
        
        self._complete_states.append(np.array(action_as_int, np.int32))
        feedback = np.array(list(self.calculate_score(action_as_string, self._code)) + [0] * (TOTAL_COLORS_TO_GUESS-2), dtype=np.int32)
        self.update_valid_actions(action_as_string, (feedback[0], feedback[1]))
        self._state = np.array([(action_as_int), np.array(feedback)], dtype=np.int32)
        
        if self._round_counter >= TOTAL_ROUNDS or self._code == action_as_string:
            self._episode_ended = True
            if self._code == action_as_string:
                self._episode_ended = True
                reward += (TOTAL_ROUNDS - self._round_counter)
            else:
                reward -= TOTAL_ROUNDS
        if self._episode_ended:
            return ts.termination(self.get_state(), reward)
        else:
            return ts.transition(self.get_state(), reward=reward, discount=1.0)