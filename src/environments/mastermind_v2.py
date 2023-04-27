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

MAX_GUESSES = TOTAL_COLORS_AMOUNT**TOTAL_COLORS_TO_GUESS
INPUT_SHAPE = (MAX_GUESSES * TOTAL_COLORS_TO_GUESS * TOTAL_COLORS_AMOUNT+ (2*TOTAL_COLORS_TO_GUESS*MAX_GUESSES),)


class MastermindEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=MAX_GUESSES-1, name='action')

        self._observation_spec = {
            'observation': BoundedArraySpec(shape=INPUT_SHAPE, dtype=np.int32, minimum=0, maximum=TOTAL_COLORS_AMOUNT-1, name='observation'),
            'valid_actions': ArraySpec(shape=(MAX_GUESSES, ), dtype=np.bool_, name="valid_actions")
        }

        self._all_actions = np.array([x for x in range(MAX_GUESSES)], np.int32)
        self._state = 0
        self._episode_ended = False
        self._step_counter = tf.Variable(0)
        self._possible_actions = []
        self._complete_states = []
        self._illegal_actions = []
        self._round_counter = 0

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

            black, white = self.get_feedback(state)
            
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

    def get_feedback(self, guess):
        """Calculates the feedback to the guess.

        Args:
            guess (array): guess by the agent

        Returns:
            tuple (x,y): feedback, where x is the number of black pegs, i.e. correct color and position, 
            and y is the number of white pegs, i.e. correct color but wrong position
        """
        black = 0
        white = 0
        tracker = [0] * TOTAL_COLORS_TO_GUESS
        guesses = [0] * TOTAL_COLORS_TO_GUESS
        actions = []

        # correct color and position
        for i in range(0, TOTAL_COLORS_TO_GUESS):
            if int(self._code[i]) == guess[i]:
                black += 1
                tracker[i] = 1
                guesses[i] = 1

        # correct color but wrong position
        for i in range(0, TOTAL_COLORS_TO_GUESS):
            if guesses[i] == 0:
                for j in range(0, TOTAL_COLORS_TO_GUESS):
                    if int(self._code[j]) == guess[i] and tracker[j] == 0 and guesses[i] == 0:
                        guesses[i] = 1
                        tracker[j] = 1
                        white += 1

        # when no color from the guess is in the right or wrong position, remove those colors from the pool
        if black == 0 and white == 0:
            for action in self._possible_actions:
                colors = self.get_action_index_as_array(action, True)
                is_illegal = False
                for g in guess:
                    if g in colors:
                        is_illegal = True
                if not is_illegal:
                    actions.append(action)
            self._possible_actions = np.array(actions)

        # if only the colors from the guess are in the pool, remove the rest
        if black + white == TOTAL_COLORS_TO_GUESS:
            for action in self._possible_actions:
                colors = self.get_action_index_as_array(action, True)
                if np.array_equal(np.sort(colors), np.sort(guess)):
                    actions.append(action)
            self._possible_actions = np.unique(np.array(actions))

        return black, white

    def update_illegal_actions(self, action):
        """Removes an action from the list of possible actions

        Args:
            action (np.array): action to delete
        """
        if action != self._code:
            self._possible_actions = np.delete(self._possible_actions, np.argwhere(self._possible_actions == action))

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

        self._state = np.zeros(shape=INPUT_SHAPE, dtype=np.int32)
        self._complete_states = []
        self._illegal_actions = []
        self._round_counter = 0

        return ts.restart(self.get_state())
    
    def one_hot(self, value):
        """One-hot encoding a color

        Args:
            value (array): color from the guess
            is_color (bool, optional): _description_. Defaults to True.

        Returns:
            array: one-hot encoded color
        """
        tmp = np.zeros(TOTAL_COLORS_AMOUNT, np.int32)
        tmp[Color(value).value] = 1
        return tmp

    def one_hot_feedback(self, feedback):
        """One-hot encoding the feedback.

        Args:
            feedback (tuple (x,y)): feedback as tuple

        Returns:
            array: one-hot encoded feedback
        """
        tmp = []
        black, white = feedback
        
        for _ in range(black):
            tmp.append([1, 0])
            
        for _ in range(white):
            tmp.append([0, 1])
        
        arr_len = len(tmp)
        if arr_len < TOTAL_COLORS_TO_GUESS:
            for _ in range(TOTAL_COLORS_TO_GUESS - arr_len):
                tmp.append([0,0])
                
        return np.array(tmp).flatten()
    
    def hamming_distance(self, string1, string2):
        if (len(string1) != len(string2)):
            raise Exception('Strings must be of equal length.')
        dist_counter = 0
        for n in range(len(string1)):
            if string1[n] != string2[n]:
                dist_counter += 1
        return dist_counter
        
    def manhatten_distance(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._step_counter.assign_add(1)
        self._round_counter += 1

        # repeated actions are prohibited
        self.update_illegal_actions(action)

        action_as_string = self.get_action_index_as_array(action)
        action_as_int = self.get_action_index_as_array(action, True)

        # add current action to total actions, for rendering purpose
        self._complete_states.append(np.array(action_as_int, np.int32))
        feedback = np.array(self.get_feedback(action_as_int), np.int32)

        s = np.hstack((
            self.one_hot(action_as_int[0]),
            self.one_hot(action_as_int[1]),
            self.one_hot(action_as_int[2]),
            self.one_hot(action_as_int[3]),
            self.one_hot_feedback(feedback),
        ))

        # set the state at the respective position on the board
        self._state[int(INPUT_SHAPE[0]*(self._round_counter-1)/MAX_GUESSES):int(INPUT_SHAPE[0]*self._round_counter/MAX_GUESSES)] = s        
        reward = -self._round_counter
        reward -= self.hamming_distance(self._code, action_as_string)

        if (feedback[0] == 0 and feedback[1] == 0) or ((feedback[0] + feedback[1]) == 4) and feedback[0] != 4:
            reward = 100 ** (1/self._round_counter)     

        if self._round_counter >= MAX_GUESSES or self._code == action_as_string:
            self._episode_ended = True
            if self._code == action_as_string:
                reward = 1500
        
        if self._episode_ended:
            return ts.termination(self.get_state(), reward)
        else:
            return ts.transition(self.get_state(), reward=reward, discount=1.0)
