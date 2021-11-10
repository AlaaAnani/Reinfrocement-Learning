# Implementing Value Iteration to Find Optimal $\pi'$ for a Karel Task $T_2$
# Documentation
There are multiple classes used to implement value iteration to find the optimal policy $\pi$. 
## bcoolors
This class is only useful for the output map coloring
## Direction
An enum structer specifying directions
## Action
An enum structer specifiyng possible actions
## ENV 
The class where all environment-related details are stored.
## MDP
This class contains all MDP-related values, implements value iteration and executes the reached policy.

The last two will be expanded on before their respective cells.

Importing Necessary libraries


```python
import numpy as np
import time
from enum import Enum
from os import system, name
from time import sleep
from IPython.display import clear_output
from ast import literal_eval as make_tuple
import random
import time
# seed random number generator
random.seed(73)
```


```python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
class Action(Enum):
    move = 0 
    turnLeft = 1 
    turnRight = 2 
    finish = 3
```

# Environemnt Class (ENV)
This class contains all environment-relataed details in a paramettrized manner. 
## ENV Attribues:
- `W` (int): width of the map
- `H`(int): height of the map
- `walls` list(): a list containing all tuple positions of the walls
- `avatar_looks` list(): contains different orientations looks indexed in-order with the `Direction` enum class.
- `map` list(list()): contains the current Karel configuragion based on the AVATAR's state.
## ENV Methods
- `make_env(self, state)`: takes the state of the avatar and generates the environemnt accordingly, then stores it in `self.map`.
- `show(self, state)`: prints the current map given the AVATAR's state.


```python
class ENV():
    def __init__(self, W=4, H=4, walls=[(1, 2), (2, 3)]):
        self.W = W
        self.H = H
        self.walls = walls
        self.avatar_looks = ['^', '>', 'v', '<']
    def make_env(self, State):
        self.map = [['.' for i in range(self.W)] for j in range(self.H)]
        for wall in self.walls:
            self.map[wall[0]][wall[1]] = '#'
        self.map[State[0]][State[1]] = self.avatar_looks[State[2]]
    def show(self, State):
        self.make_env(State)
        for r in range(self.W):
            for c in range(self.H):
                if self.map[r][c] in self.avatar_looks:
                    print(bcolors.OKBLUE + self.map[r][c] + bcolors.ENDC," ", end='')
                elif self.map[r][c] == 'm':
                    print(bcolors.WARNING + 'm' + bcolors.ENDC," ", end='')
                elif self.map[r][c] == '#':
                    print(bcolors.FAIL + '#' + bcolors.ENDC," ", end='') 
                else:
                    print(self.map[r][c], " ", end='')
            print('\n')
```

# Makrov-Decision Process Class (MDP)
This class contain all MDP-related dynamics.
## MDP Attributes
- `initial_state`: a 3-tuple containg the intial state of task $T_1$, which is `(1, 2, Direction.LEFT.value)`.
- `final_state`: a 3-tuple represeing the post-grid configuration, which is `(3, 2, Direction.RIGHT.value)`
- `env`: an `ENV()` class instance
- `S`: a list of 3-tuples containing all possible states, $(i, j, dir)$ such that:  

>>> $0 \leq i \leq HEIGHT -1$  

>>> $0 \leq j \leq WIDTH - 1 $  

>>> $dir \in \{UP, RIGHT, DOWN, LEFT\}$

>>> it also include the terminal state, namely, `terminal`

- `A`: a dictionary of $S$, which contains all possible actions given $S$. This is $A(S)$
- `V`: a dictionary of $S$, which represent the value function of $S$. This is $V(S)$
## MDP Methods
- `generateState(self):` generates all possible states and assigns them to the attribute `S`.
- `generateActions(self):` generates $A(S)$ and assigns them to $A$.
- ` generateV(self)`: $\forall_{s\in S} V(S) \gets 0$
- `reward(s, a)`: takes $s$ and $a$, returns $100$ if the state is the desired post-grid and $a=finish$. Returns -1, otherwise.
- `p(s, a)`: represents the environment dynamics which takes $s$ and $a$ and returns the next state $s'$ and immediate reward $r$.
- `ValueIteration(theta, V, S, A, gamma=0.97)`: implements the value iteration algorithm in the textbook (pg: 105), returns the reached optimal policy $\pi'$.
- `executePolicy(self, pi)`: given $\pi$, and class attributes such as $S$, $A(S)$ and $V(S)$, executes the policy $\pi$ while using `self.env` for visualization of the execution.


```python
class MDP():
    def __init__(self):
        self.initial_state = (1, 1, Direction.LEFT.value)
        self.final_state = (3, 2, Direction.RIGHT.value)
        self.env = ENV()
        self.S = self.generateState()
        self.A = self.generateActions()
        self.V = self.generateV()

    def generateState(self):
        DIRECTIONS = [Direction.UP.value, Direction.RIGHT.value, Direction.DOWN.value, Direction.LEFT.value] 
        # Generate all possible states
        S = []
        for i in range(self.env.W):
            for j in range(self.env.H):
                for dir in DIRECTIONS:
                    S.append((i, j, dir))
        S.append('terminal')
        return S

    def generateActions(self):
        # generate all possible actions given every state
        A = {}
        for s in S:
            A[s] = []
            if s == 'terminal':
                continue
            # these actions are possible from all states
            A[s].append(Action.turnLeft.value)
            A[s].append(Action.turnRight.value)
            A[s].append(Action.finish.value)
            A[s].append(Action.move.value)
        return A
        
    def generateV(self):
        # generate V(s)
        V = {}
        for s in S:
            V[s] = 0
        # set terminal state value to 0
        V['terminal'] = 0
        return V
        
    @staticmethod
    def reward(s, a):
        if s == (3, 2, Direction.RIGHT.value) and a == Action.finish.value:
            return 100
        return -1
    @staticmethod
    def p(s, a):
        r = MDP().reward(s, a)
        # get s'
        i, j, dir = s
        new_i = i
        new_j = j
        new_dir = dir
        terminal = False
        WALL1 = (1, 2)
        WALL2 = (2, 3)
        if a == Action.move.value:
            if dir == Direction.UP.value:
                new_i -= 1
            elif dir == Direction.RIGHT.value:
                new_j += 1
            elif dir == Direction.DOWN.value:
                new_i +=1
            elif dir == Direction.LEFT.value:
                new_j -=1
        elif a == Action.turnRight.value:
            if dir == Direction.UP.value:
                new_dir = Direction.RIGHT.value
            elif dir == Direction.RIGHT.value:
                new_dir = Direction.DOWN.value
            elif dir == Direction.DOWN.value:
                new_dir = Direction.LEFT.value
            elif dir == Direction.LEFT.value:
                new_dir = Direction.UP.value
        elif a == Action.turnLeft.value:
            if dir == Direction.UP.value:
                new_dir = Direction.LEFT.value
            elif dir == Direction.RIGHT.value:
                new_dir = Direction.UP.value
            elif dir == Direction.DOWN.value:
                new_dir = Direction.RIGHT.value
            elif dir == Direction.LEFT.value:
                new_dir = Direction.DOWN.value
        elif a == Action.finish.value:
            terminal = True
        
        out_of_bounds = new_i < 0 or new_i > HEIGHT-1 or new_j <0 or new_j > WIDTH - 1
        on_a_wall = (new_i, new_j) == WALL1 or (new_i, new_j) == WALL2
        if terminal or out_of_bounds or on_a_wall:
            s_ = 'terminal'
        else:
            s_ = (new_i, new_j, new_dir)
        return r, s_

    @staticmethod  
    def ValueIteration(theta, V, S, A, gamma=0.97):
        # theta = 1e-3
        Delta = 1e9
        num_iterations = 0
        while(Delta >= theta):
            Delta = 0
            for s in S:
                v = V[s]
                max_a = -1e9
                for a in A[s]:
                    r, s_ = MDP().p(s, a)
                    curr_v = r + gamma*V[s_]
                    if curr_v > max_a:
                        V[s] = curr_v
                        max_a = curr_v
                Delta = max(Delta, np.abs(v-V[s]))
            num_iterations += 1
        # deterministic pollicy pi
        pi = {}
        for s in S:
            max_a = -1e9
            a_max_i = 0
            # get values all possible actions from s
            for i, a in enumerate(A[s]):
                r, s_ = MDP().p(s, a)
                curr_v = r + gamma*V[s_]
                if curr_v > max_a:
                    a_max = a
                    a_max_i = i
                    max_a = curr_v
            
            pi[s] = a_max_i
        print("Number of iterations=", num_iterations)
        return pi    
    
    def executePolicy(self, pi):
        s = self.initial_state
        action_sequence = []
        while(s != 'terminal'):
            time.sleep(2)
            clear_output(wait=True)
            self.env.show(s)
            # get action
            a = A[s][pi[s]]
            r, s_ = MDP().p(s, a)
            s = s_
            action_sequence.append(a)
        return action_sequence
        
```


```python
# generate MDP() instance, run value iteration
mdp = MDP()
start = time.time()
pi = MDP().ValueIteration(1e-3, mdp.V, mdp.S, mdp.A, gamma=0.9)
end = time.time()
print("Total runtime to learn the optimal policy=", end-start, "(s)")

```

    Number of iterations= 8
    Total runtime to learn the optimal policy= 0.6183204650878906 (s)
    


```python
# policy execution
action_sequence = mdp.executePolicy(pi)
```

    .  .  .  .  
    
    .  .  [91m#[0m  .  
    
    .  .  .  [91m#[0m  
    
    .  .  [94m>[0m  .  
    
    


```python
# printing action commands
actions = ['move', 'turnLeft', 'turnRight', 'finish']
print("Actions sequence from optimal policy:")
for a in action_sequence:
    print(actions[a])
```

    Actions sequence from optimal policy:
    turnLeft
    move
    move
    turnLeft
    move
    finish
    
