import torch
import torch.nn as nn
import numpy as np

DEV = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float
LEARN_RATE = 1e-3

P1 = 1
P2 = -1
TIE = 0

class ggpNN(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(ggpNN, self).__init__()
        self.ggp_nn = nn.Sequential(
            nn.Linear(inp_dim, inp_dim * 10),
            nn.Sigmoid(),
            nn.Linear(inp_dim * 10, inp_dim * 15),
            nn.Sigmoid(),
            nn.Linear(inp_dim * 15, out_dim),
            nn.Sigmoid(),
            nn.Softmax(dim = 1)
        )
        self.state_label_map = {}
    
    def fwd_prop(self, x):
        return self.ggp_nn(x)

    def get_arg_max(self, nn_op):
        max_val = -1
        max_ind = -1
        index = 0
        for prob in nn_op:
            if prob > max_val:
                max_val = prob
                max_ind = index
            index += 1
        return max_ind

    def get_node_state(self, board, player_to_move):
        state = []

        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                state.append(board[i, j])

        state.append(player_to_move)
        state = [state]
        state = torch.tensor(state, device = DEV, dtype = DTYPE)

        return state

    def get_node_label(self, node):
        node_label = []
        max_ind = -1
        max_val = -1
        index = 0

        for child in node.children:
            if child != None:
                prob_i = 0
                if child.sample_count > 0:
                    prob_i = child.value/child.sample_count
                if prob_i > max_val:
                    max_val = prob_i
                    max_ind = index
            node_label.append(0)
            index += 1

        node_label[max_ind] = 1
        return node_label
        
    def train_rollout_network(self, root, actions, t_iterations):
        node = root
        node_labels = []
        node_data = []
        player = P1

        for action in actions:
            node_label = self.get_node_label(node)
            node_state = self.get_node_state(node.state, player)
            
            self.state_label_map[tuple(node_state[0])] = node_label

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node=None

            player = player*-1

        for data in self.state_label_map.keys():
            node_data.append(list(data))
            node_labels.append(self.state_label_map[data])

        node_labels = torch.tensor(node_labels, device=DEV, dtype=DTYPE)
        node_data = torch.tensor(node_data, device=DEV, dtype=DTYPE)

        loss_fn = nn.MSELoss()
        optimizer=torch.optim.Adam(self.ggp_nn.parameters(), lr=LEARN_RATE)

        for i in range(t_iterations):
            res = self.fwd_prop(node_data)
            loss = loss_fn(res, node_labels)
            if i%100==0:
                print("Loss at %d: %f"%(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()