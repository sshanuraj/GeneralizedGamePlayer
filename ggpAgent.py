import numpy as np
import random
from ggBoard import ggBoard
from ggNode import ggNode

P1 = 1
P2 = -1
TIE = 0

WIN_REWARD_TERMINAL = 2
LOSS_REWARD_TERMINAL = -2
TIE_REWARD_TERMINAL = 0

WIN_REWARD = 1
LOSS_REWARD = -1
TIE_REWARD = 0

class ggpAgent:
    def __init__(self, player):
        self.player = player

    def get_reward_terminal(self, player_to_move, win_player, move_count):
        if win_player == player_to_move:
            return LOSS_REWARD_TERMINAL
        
        if win_player == TIE:
            return TIE_REWARD_TERMINAL
        
        return WIN_REWARD_TERMINAL
    
    def get_reward(self, player_to_move, win_player, move_count):
        if win_player == player_to_move:
            return LOSS_REWARD
        
        if win_player == TIE:
            return TIE_REWARD
        
        return WIN_REWARD

    def get_final_reward(self, root, actions, player, board):
        node = root
        prev_node = root
        ini_player = P2

        for action in actions:
            prev_node = node

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node = None
            ini_player = self.switch_player(ini_player)
            if not node:
                prev_node.expand_node(board, ini_player)
                node = prev_node.children[action]
        ini_player = self.switch_player(ini_player)
        reward = self.get_reward(ini_player, player, board.move_count)
        #print(root.value, root.sample_count)
        node.backpropagate(reward)
        #print(root.value, root.sample_count)
        return root

    def get_move_coords(self, move, is_drop, cols, dimensions):
        x = 0
        y = 0

        if is_drop:
            x = cols[move]
            y = move
        else:
            x = move // dimensions[1]
            y = move % dimensions[1]

        return x, y

    def make_rand_virtual_move(self, ggp_nn, player, state, dimensions, is_drop, cols, win_num):
        action = -1
        board = ggBoard(dimensions, win_num, is_drop)
        
        possible_moves = []
        if is_drop:
            for i in range(dimensions[1]):
                if cols[i] >= 0:
                    possible_moves.append(i)
        else:
            for i in range(dimensions[0] * dimensions[1]):
                x, y = self.get_move_coords(i, is_drop, cols, dimensions)
                if state[x][y] == 0:
                    possible_moves.append(i)

        # #check for if win possible in next move
        winning_move = board.check_for_winning_move(state, cols, player)
        if winning_move != -1:
            x, y = self.get_move_coords(winning_move, is_drop, cols, dimensions)
            last_move = [x, y]
            # print(winning_move, last_move)
            state[x][y] = player
            if is_drop:
                cols[winning_move] -= 1
            return state, cols, last_move

        # #checking loss to avoid, first loss potential found will be used as move
        player = self.switch_player(player)
        win_move_for_opp = board.check_for_winning_move(state, cols, player)
        player = self.switch_player(player)
        if win_move_for_opp != -1:
            x, y = self.get_move_coords(win_move_for_opp, is_drop, cols, dimensions)
            last_move = [x, y]
            # print(win_move_for_opp, last_move)
            state[x][y] = player
            if is_drop:
                cols[win_move_for_opp] -= 1
            return state, cols, last_move

        
        #no win found and no loss potential found, continue as normal with random playout
        l = len(possible_moves)

        action = -1
        if np.random.rand() < 0.6:  #60-40 split bw using rollout network or random move
            ret = ggp_nn.fwd_prop(ggp_nn.get_node_state(state, player))
            action = ggp_nn.get_arg_max(ret[0])
            if is_drop:
                if cols[action] < 0:
                    action = np.random.randint(0, l)
                    action = possible_moves[action]
            else:
                x, y = self.get_move_coords(action, is_drop, cols, dimensions)
                if state[x][y] != 0:
                    action = np.random.randint(0, l)
                    action = possible_moves[action]
        else:
            action = np.random.randint(0, l)
            action = possible_moves[action]
    
        last_move = []
        
        x, y = self.get_move_coords(action, is_drop, cols, dimensions)
        state[x][y] = player
        last_move = [x, y]
        if is_drop:
            cols[action] -= 1

        return state, cols, last_move
    
    def switch_player(self, player):
        return player*-1
    
    def rollout(self, ggp_nn, state, cols, is_drop, win_num, move_count, player_move, dimensions):
        board = ggBoard(dimensions, win_num, is_drop)
        MAX_MOVES = dimensions[0] * dimensions[1]
        player_to_move = player_move
        ini_move_count = move_count
        while True:
            state, cols, last_move = self.make_rand_virtual_move(ggp_nn, player_move, state, dimensions, is_drop, cols, win_num)
            move_count += 1

            if board.check_win_virtual(state, last_move):
                return self.get_reward(player_to_move, player_move, ini_move_count)

            if move_count == MAX_MOVES:
                return TIE_REWARD
            
            player_move = self.switch_player(player_move)

    def get_best_move(self, ggp_nn, actions, n_iterations, root, board, train_ = True):
        action = 0
        count = 0 
        node = root
        prev_node = root
        player = P2
        next_node = None

        for action in actions:
            prev_node = node

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node = None
            player = self.switch_player(player)
            if not node: 
                prev_node.expand_node(board, player)
                node = prev_node.children[action]
            

        if node.check_leaf():
            node.expand_node(board, self.player) 

        curr = node
        change = False

        while count < n_iterations:
            if not change: 
                curr = node
            if curr.check_leaf():
                if curr.sample_count == 0:
                    #rollout
                    if curr.is_terminal:
                        player_to_move = P2 if curr.move_count%2 == 1 else P1
                        reward = self.get_reward_terminal(player_to_move, curr.win_color, curr.move_count)
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                    else:
                        vstate = curr.state.copy()
                        vcols = curr.cols.copy()
                        player_to_move = P2 if curr.move_count%2 == 1 else P1
                        
                        reward = self.rollout(ggp_nn, vstate, vcols, board.is_drop, board.win_num, curr.move_count, player_to_move, board.dimensions)
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                else:
                    player_to_move = P2 if curr.move_count%2 == 1 else P1

                    if curr.is_terminal:
                        reward = self.get_reward_terminal(player_to_move, curr.win_color, curr.move_count)
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    curr.expand_node(board, player_to_move)

                    curr, _, _, _ = curr.get_max_node(root.sample_count, train_)

                    if curr.is_terminal:
                        player_to_move = P2 if curr.move_count%2 == 1 else P1
                        # print("is terminal node after expansion")
                        reward = self.get_reward_terminal(player_to_move, curr.win_color, curr.move_count)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    vstate = curr.state.copy()
                    vcols = curr.cols.copy()

                    player_to_move = P2 if curr.move_count%2 == 1 else P1

                    reward = self.rollout(ggp_nn, vstate, vcols, board.is_drop, board.win_num, curr.move_count, player_to_move, board.dimensions)
                    curr.backpropagate(reward)
                    
                    count += 1
                    change = False
                    continue

            else:
                change = True
                curr, _ , _, _= curr.get_max_node(root.sample_count, train_)

        next_node, action, ucbs, sample = node.get_max_node(root.sample_count, train_)

        f_ucbs = np.array(ucbs)
        fo_ucbs = []
        for ucb in f_ucbs:
            if ucb is None:
                fo_ucbs.append(None)
            else:
                fo_ucbs.append("{:.2f}".format(ucb))
        print("UCB1 Values for the current move:", fo_ucbs)
        if train_:
            print(sample)

        #check for direct win
        win_index = board.check_for_winning_move(node.state, node.cols, self.player)
        if win_index != -1:
            return root, win_index, ucbs

        #check for loss potential
        loss_index = board.check_for_winning_move(node.state, node.cols, self.switch_player(self.player))
        if loss_index != -1:
            return root, loss_index, ucbs

        return root, action, ucbs