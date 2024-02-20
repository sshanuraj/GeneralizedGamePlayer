import random
import numpy as np

class ggBoard:
    def __init__(self, dimensions, win_num, is_drop):
        self.dimensions = dimensions
        self.board = np.zeros(dimensions)
        self.x = self.board.shape[0]
        self.y = self.board.shape[1]
        self.move_count = 0
        self.cols = [self.x - 1]*self.y  #to keep track of "dropping" moves
        self.win_num = win_num
        self.is_drop = is_drop
        self.last_move = [-1, -1]
        self.max_moves = self.x * self.y
    
    def get_board_to_state(self, board):
        pass

    def in_boundary(self, x, y):
        if x >= self.x or x < 0 or y >= self.y or y < 0:
            return False
        return True

    def is_move_possible(self, move_index):
        if self.is_drop:
            if self.cols[move_index] < 0:
                return False
        else:
            x = move_index // self.y
            y = move_index % self.y
            if self.board[x][y] != 0:
                return False
        return True

    def make_move(self, player, move):
        if self.is_drop:
            self.board[self.cols[move]][move] = player
            self.last_move = [self.cols[move], move]
            self.cols[move] -= 1
        else:
            move_x = move // self.y
            move_y = move % self.y
            self.board[move_x][move_y] = player
            self.last_move = [move_x, move_y]

        self.move_count += 1

    def show_board(self):
        dis_dict = {0 : ".", 1 : "1", -1 : "2"}

        for i in range(self.x):
            for j in range(self.y):
                point = dis_dict[self.board[i][j]]
                print(point, end=" ")
            print()
        print()
        
    def reset_board(self):
        self.board = np.zeros(self.dimensions)
        self.move_count = 0
        self.last_move = [-1, -1]
        self.cols = [self.x - 1]*self.y  

    def check_down(self, state, last_move):
        x = last_move[0]
        y = last_move[1]
        color = state[x][y] 
        count = 0

        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            x += 1
            count += 1
            if count == self.win_num:
                return True

        x = last_move[0] - 1
        y = last_move[1]

        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                return False
            x -= 1
            count += 1
            if count == self.win_num:
                return True
        return False

    def check_for_winning_move(self, state, cols, player):
        bcopy = state.copy()
        if self.is_drop:
            for i in range(self.y):
                if cols[i] >= 0:
                    bcopy[self.cols[i]][i] = player
                    if self.check_win_virtual(bcopy, [self.cols[i], i]):
                        return i
                    bcopy[self.cols[i]][i] = 0
        else:
            MAX_MOVES = self.x * self.y
            for i in range(MAX_MOVES):
                x = i // self.y
                y = i % self.y
                if bcopy[x][y] == 0:
                    bcopy[x][y] = player
                    if self.check_win_virtual(bcopy, [x, y]):
                        return i
                    bcopy[x][y] = 0
        return -1
    
    def check_horizontal(self, state, last_move):
        x = last_move[0]
        y = last_move[1]
        color = state[x][y]
        count = 0
        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            y += 1
            count += 1
            if count == self.win_num:
                return True

        x = last_move[0]
        y = last_move[1] - 1

        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            y -= 1
            count += 1
            if count == self.win_num:
                return True

        if count >= self.win_num:
            return True
        return False

    def check_diag(self, state, last_move):
        x = last_move[0]
        y = last_move[1]
        color = state[x][y]
        count = 0
        # upright 
        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            x -= 1
            y += 1
            count += 1
            if count == self.win_num:
                return True

        #downleft
        x = last_move[0] + 1
        y = last_move[1] - 1
        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            x += 1
            y -= 1
            count += 1
            if count == self.win_num:
                return True

        #upleft
        x = last_move[0]
        y = last_move[1]
        count = 0
        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            x -= 1
            y -= 1
            count += 1
            if count == self.win_num:
                return True

        #down right
        x = last_move[0] + 1
        y = last_move[1] + 1
        while True:
            if not self.in_boundary(x, y) or color != state[x][y]:
                break
            x += 1
            y += 1
            count += 1
            if count == self.win_num:
                return True
        return False

    def check_win_virtual(self, state, last_move):
        if self.check_down(state, last_move):
            return True
        if self.check_diag(state, last_move):
            return True
        if self.check_horizontal(state, last_move):
            return True
        return False

    def check_virtual_draw(self, state):
        if 0 in state:
            return False
        return True

    def check_win(self):
        return self.check_win_virtual(self.board, self.last_move)


# b = ggBoard((6,7), 4, False)

# b.make_move(1, 41)
# b.make_move(-1, 29)

# b.show_board()

