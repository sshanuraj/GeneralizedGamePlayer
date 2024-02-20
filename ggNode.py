import random
import numpy as np
from ggBoard import ggBoard

INF = 1e10

P1 = 1
P2 = -1
TIE = 0

class ggNode:
    def __init__(self, parent, num_actions, state, cols, move_count, dimensions):
        self.value = 0
        self.sample_count = 0
        self.parent = parent
        self.children = []
        self.num_actions = num_actions
        self.dimensions = dimensions

        self.state = state
        self.cols = cols
        self.move_count = move_count

        self.is_terminal = False
        self.win_color = 0

    def show_params(self):
        avg_value = "INF"
        if self.sample_count != 0:
            avg_value = str(self.value / self.sample_count)
        sc = str(self.sample_count)

        print("Average value: %s"%(avg_value))
        print("Visit count: %s"%(sc))
        print("Node state: %s"%(str(self.is_terminal)))
        print("Move count: " + str(self.move_count))
        print("Win color: "+str(self.win_color))
        print("Child states: ")
        if self.is_terminal:
            print("Empty")
        else:
            for i in self.children:
                if i == None:
                    print(i)
                else:
                    print(i.state)

    def go_up(self):
        return self.parent

    def go_to(self, c_index):
        if self.children == []:
            return None
        return self.children[c_index]
    
    def check_leaf(self):
        if len(self.children) == 0:
            return True
        return False

    def calculate_ucb(self, N, c): 
        if self.sample_count == 0:
            return INF
        return (self.value/self.sample_count) + (c*(np.log(self.parent.sample_count)/self.sample_count)**0.5)

    def get_softmax_array(self, arr, check_zero = False):
        exp_arr = []
        for val in arr:
            if check_zero:
                if val == 0:
                    exp_arr.append(0)
                else:
                    exp_arr.append(np.exp(val))
            elif val != None:
                exp_arr.append(np.exp(val))
            else:
                exp_arr.append(0)
        exp_sum = np.sum(exp_arr)
        #print(arr)
        exp_arr = exp_arr/exp_sum
    
        return exp_arr

    def get_max_node(self, N, train_ = True):
        ucbs = []

        if self.is_terminal:
            return None, -1, [], []

        inc = 0

        for node in self.children:
            if node:
                ucbs.append(node.calculate_ucb(N, 1.414))
            else:
                ucbs.append(None)

        max_val = 0 - INF
        max_inds = []
        max_ind = -1
        sample = []
        l = len(self.children)
        if train_:
            if INF not in ucbs:
                ucbs = self.get_softmax_array(ucbs)
                #print(ucbs)
                sample = np.random.multinomial(1000, ucbs, size=1)
                max_ind = np.argmax(sample)
            else:
                for i in range(l):
                    if ucbs[i] != None and ucbs[i] == max_val:
                        max_inds.append(i)
                    if ucbs[i] != None and ucbs[i] > max_val:
                        max_inds = [i]
                        max_val = ucbs[i]
                max_ind = max_inds[np.random.randint(0, len(max_inds))]
        else:
            for i in range(l):
                if ucbs[i] != None and ucbs[i] == max_val:
                    max_inds.append(i)
                if ucbs[i] != None and ucbs[i] > max_val:
                    max_inds = [i]
                    max_val = ucbs[i]
            max_ind = max_inds[np.random.randint(0, len(max_inds))]

        max_node = self.children[max_ind]
        return max_node, max_ind, ucbs, sample

    def get_min_node(self, N): 
        ucbs = []

        if self.is_terminal:
            return None, -1, []

        for node in self.children:
            if node:
                ucbs.append(node.calculate_ucb(N, 2))
            else:
                ucbs.append(None)

            min_ind = 0
            min_val = 0 - INF
            l = len(self.children)

            for i in range(l):
                if ucbs[i] != None and ucbs[i] < min_val:
                    min_ind = i
                    min_val = ucbs[i]

            min_node = self.children[min_ind]
            return min_node, min_ind, ucbs
        
    def backpropagate(self, reward):
        self.sample_count += 1
        self.value += reward
        curr = self.parent
        alpha = 0.97
        reward = reward*alpha
        while curr:
            curr.sample_count += 1
            reward = -1*reward #alternate between negative and positive reward accoeding to the player to move
            curr.value += reward
            curr = curr.go_up()
            reward = reward * alpha
    
    def expand_node(self, board, player):
        if self.is_terminal:
            return None

        MAX_MOVES = board.max_moves
        for move_index in range(self.num_actions):
            # print("here")
            next_state = 0
            last_move = []
            cols = self.cols.copy()
            if board.is_drop:
                if cols[move_index] <= -1: #check if valid move 
                    self.children.append(None)
                    continue
                next_state = self.state.copy()  #copying next state for child node
                next_state[cols[move_index]][move_index] = player  #making move for child node state
                last_move = [cols[move_index], move_index]
                cols[move_index] -= 1

            else: #not dropping the pieces
                i = move_index // board.y
                j = move_index % board.y
                if self.state[i][j] == 0:
                    next_state = self.state.copy()
                    next_state[i][j] = player
                    last_move = [i, j]
                else:
                    self.children.append(None)
                    continue

            node = ggNode(self, self.num_actions, next_state, cols, self.move_count + 1, self.dimensions)

            if board.check_win_virtual(next_state, last_move):
                node.is_terminal = True
                node.win_color = player #win for P1/P2

            elif node.move_count == MAX_MOVES:
                node.is_terminal = True
                node.win_color = TIE

            self.children.append(node)