import numpy as np
import random
import math
from ggBoard import ggBoard
from ggNode import ggNode
from ggpAgent import ggpAgent 
from ggpNN import ggpNN
from ggpUI import GGP_UI

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')
plt.ion()

P1 = 1
P2 = -1
TIE = 0

style.use('fivethirtyeight')
plt.ion()

class GGP:
    def __init__(self, game_root, board, game):
        self.game_root = game_root
        self.actions = []
        self.board = board
        self.game = game
        #self.game_root.expand_node(board, P1)
        self.p1_agent = ggpAgent(P1)
        self.p2_agent = ggpAgent(P2)

        inp_dim = self.board.max_moves + 1
        out_dim = self.game_root.num_actions
        self.ggp_nn = ggpNN(inp_dim, out_dim)

        self.ucb_p1 = []
        self.ucb_p2 = []

        self.ui = GGP_UI(30, (50*board.y, 50*board.x), board.dimensions)
    
    def write_games(self, actions, winner, game_format):
        f=open("game_rep.txt", "a")
        f.write("\n----NEW GAME DETAILS----\n")
        f.write("Game Name: %s\n"%(self.game))
        f.write(str(actions)+"\n")
        f.write("UCB Values for P1:" + str(self.ucb_p1) + "\n")
        f.write("UCB Values for P2:"+ str(self.ucb_p2) + "\n")
        f.write("Winner: %s\n"%(winner))
        f.write("Format: %s\n"%(game_format))
        f.write("----GAME DETAILS END----\n")
        f.close()

    
    def play_train(self, num_games, num_iterations, train_iterations):
        print("----- %s -----"%(self.game))
        w1 = 0
        w2 = 0
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        while self.ui.con:
            for i in range(num_games):
                win = False
                print("-----  GAME %s  -----\n"%(str(i+1)))
                actions = []
                MAX_MOVES = self.board.max_moves
                
                for j in range(MAX_MOVES):
                    self.ui.setBoard()
                    self.ui.checkGameQuitState()
                    if j%2 == 0:  #P1 move
                        self.game_root, action, ucbs = self.p1_agent.get_best_move(self.ggp_nn, actions, num_iterations, self.game_root, self.board, False)
                        actions.append(action)
                        self.board.make_move(P1, action)
                        print("Player %d played move %d"%(1, action))
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P1)
                        self.ui.setBoard()
                        ucb_avg = self.get_ucbs_avg(ucbs)
                        self.ucb_p1.append(ucb_avg)
                        if self.board.check_win():
                            print("Player 1 WINS\n")
                            #self.game_root = self.p1_agent.get_final_reward(self.game_root, actions, P1, self.board)
                            self.write_games(actions, "P1", "P1vsP2")
                            win = True
                            w1 += 1
                            break
                    else: #P2 move
                        self.game_root, action, ucbs = self.p2_agent.get_best_move(self.ggp_nn, actions, num_iterations, self.game_root, self.board, False)
                        actions.append(action)
                        self.board.make_move(P2, action)
                        print("Player %d played move %d"%(2, action))
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P2)
                        self.ui.setBoard()
                        ucb_avg = self.get_ucbs_avg(ucbs)
                        self.ucb_p2.append(ucb_avg)
                        if self.board.check_win():
                            print("Player 2 WINS\n")
                            #self.game_root = self.p2_agent.get_final_reward(self.game_root, actions, P2, self.board)
                            self.write_games(actions, "P2", "P1vsP2")
                            win = True
                            w2 += 1
                            break	
                    
                    ax1.clear()
                    plt.title("Evaluations per Move")
                    plt.ylabel("Potential Reward")
                    plt.xlabel("Move count")
                    plt.ylim((-2.2, 2.2))
                    ax1.plot(np.arange(len(self.ucb_p1))+1, self.ucb_p1)
                    ax1.plot(np.arange(len(self.ucb_p2))+1, self.ucb_p2)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    self.ui.setBoard()
                    self.ui.displayScreen()
                    self.ui.clockTick()
                if not win:
                    print("DRAW\n")
                    #self.game_root = self.p1_agent.get_final_reward(self.game_root, actions, TIE, self.board)
                    self.write_games(actions, "DRAW", "P1vsP2")
                    w1 += 0.5
                    w2 += 0.5
                
                self.clear_ucbs()
                print("-----  GAME %s ENDS  -----\n"%(str(i+1)))
                if i%2 == 0:
                    self.ggp_nn.train_rollout_network(game_root, actions, train_iterations)
                self.ui.displayScreen()
                self.ui.clockTick()
                self.ui.clearMoves()		
                self.board.reset_board()
            self.ui.endScreen()
        print("Results: Player 1: %.1f, Player 2: %.1f"%(w1, w2))
    
    def clear_ucbs(self):
        self.ucb_p1 = []
        self.ucb_p2 = []

    def play_against_p1(self, num_games, num_iterations):
        print("----- %s -----"%(self.game))
        w1 = 0
        w2 = 0
        num_moves = 0
        if self.board.is_drop:
            num_moves = self.board.y
        else:
            num_moves = self.board.max_moves
        self.ui.initScreen()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        while self.ui.con:
            for i in range(num_games):
                win = False
                print("-----  GAME %s  -----\n"%(str(i+1)))
                actions = []
                MAX_MOVES = self.board.max_moves
                for j in range(MAX_MOVES):
                    self.ui.setBoard()
                    self.ui.checkGameQuitState()
                    if j%2 == 0:  #P1 move
                        self.game_root, action, ucbs = self.p1_agent.get_best_move(self.ggp_nn, actions, num_iterations, self.game_root, self.board, False)
                        actions.append(action)
                        self.board.make_move(P1, action)
                        print("Player %d played move %d"%(1, action))
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P1)
                        self.ui.setBoard()
                        ucb_avg = self.get_ucbs_avg(ucbs)
                        self.ucb_p1.append(ucb_avg)
                        if self.board.check_win():
                            print("Player 1 Wins\n")
                            #self.game_root = self.p1_agent.get_final_reward(self.game_root, actions, P1, self.board)
                            self.write_games(actions, "P1", "P1vsHuman")
                            win = True
                            w1 += 1
                            break
                    else: #P2 move
                        action = 0
                        while True:
                            action = input("Enter move:")
                            if action.isdigit():
                                action = int(action)
                            else:
                                print("Not an integer move index.")
                                continue

                            if action >= num_moves or action < 0:
                                print("Wrong move index... 0 - %d"%(num_moves - 1))
                            elif not self.board.is_move_possible(action):
                                print("Wrong move index.. piece placed already.")
                            else:
                                break
                        actions.append(action)
                        self.board.make_move(P2, action)
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P2)
                        self.ui.setBoard()
                        if self.board.check_win():
                            print("Player 2 Wins\n")
                            #self.game_root = self.p1_agent.get_final_reward(self.game_root, actions, P2, self.board)
                            self.write_games(actions, "Human", "P1vsHuman")
                            win = True
                            w2 += 1
                            break
                    ax1.clear()
                    plt.title("Evaluations per Move")
                    plt.ylabel("Potential Reward")
                    plt.xlabel("Move count")
                    plt.ylim((-2.2, 2.2))
                    ax1.plot(np.arange(len(self.ucb_p1))+1, self.ucb_p1)
                    fig.canvas.draw()
                    fig.canvas.flush_events()	
                    self.ui.setBoard()
                    self.ui.displayScreen()
                    self.ui.clockTick()
                if not win:
                    print("DRAW\n")
                    #self.game_root = self.p1_agent.get_final_reward(self.game_root, actions, TIE, self.board)
                    self.write_games(actions, "DRAW", "P1vsHuman")
                    w1 += 0.5
                    w2 += 0.5
                print("-----  GAME %s ENDS  -----\n"%(str(i+1)))		
                self.ui.displayScreen()
                self.ui.clockTick()
                self.ui.clearMoves()
                self.clear_ucbs()
                self.board.reset_board()
            self.ui.endScreen()
        print(w1, w2)
    
    def play_against_p2(self, num_games, num_iterations):
        print("----- %s -----"%(self.game))
        w1 = 0
        w2 = 0
        num_moves = 0
        if self.board.is_drop:
            num_moves = self.board.y
        else:
            num_moves = self.board.max_moves

        self.ui.initScreen()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        while self.ui.con:
            for i in range(num_games):
                win = False
                print("-----  GAME %s  -----\n"%(str(i+1)))
                actions = []
                MAX_MOVES = self.board.max_moves
                for j in range(MAX_MOVES):
                    self.ui.setBoard()
                    self.ui.checkGameQuitState()
                    if j%2 == 0:  #P1 move
                        action = 0
                        while True:
                            action = input("Enter move:")
                            if action.isdigit():
                                action = int(action)
                            else:
                                print("Not an integer move index.")
                                continue
                            if action >= num_moves or action < 0:
                                print("Wrong move index... 0 - %d"%(num_moves - 1))
                            elif not self.board.is_move_possible(action):
                                print("Wrong move index.. piece placed already.")
                            else:
                                break
                        actions.append(action)
                        self.board.make_move(P1, action)
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P1)
                        self.ui.setBoard()
                        if self.board.check_win():
                            print("Player 1 Wins\n")
                            #self.game_root = self.p2_agent.get_final_reward(self.game_root, actions, P1, self.board)
                            self.write_games(actions, "Human", "HumanvsP2")
                            win = True
                            w1 += 1
                            break
                        
                    else: #P2 move
                        self.game_root, action, ucbs = self.p2_agent.get_best_move(self.ggp_nn, actions, num_iterations, self.game_root, self.board, False)
                        actions.append(action)
                        self.board.make_move(P2, action)
                        print("Player %d played move %d"%(2, action))
                        self.board.show_board()
                        self.ui.addMove(self.board.last_move, P2)
                        self.ui.setBoard()
                        ucb_avg = self.get_ucbs_avg(ucbs)
                        self.ucb_p2.append(ucb_avg)
                        if self.board.check_win():
                            print("Player 2 Wins\n")
                            #self.game_root = self.p2_agent.get_final_reward(self.game_root, actions, P2, self.board)
                            self.write_games(actions, "P2", "HumanvsP2")
                            win = True
                            w2 += 1
                            break	
                    ax1.clear()
                    plt.title("Evaluations per Move")
                    plt.ylabel("Potential Reward")
                    plt.xlabel("Move count")
                    plt.ylim((-2.2, 2.2))
                    ax1.plot(np.arange(len(self.ucb_p1))+1, self.ucb_p1)
                    ax1.plot(np.arange(len(self.ucb_p2))+1, self.ucb_p2)
                    fig.canvas.draw()
                    fig.canvas.flush_events()	
                    self.ui.setBoard()
                    self.ui.displayScreen()
                    self.ui.clockTick()
                if not win:
                    print("DRAW\n")
                    #self.game_root = self.p2_agent.get_final_reward(self.game_root, actions, TIE, self.board)
                    self.write_games(actions, "DRAW", "HumanvsP2")
                    w1 += 0.5
                    w2 += 0.5
                self.ui.setBoard()
                self.ui.displayScreen()
                self.ui.clockTick()
                self.ui.clearMoves()
                self.clear_ucbs()
                print("-----  GAME %s ENDS  -----\n"%(str(i+1)))		
                self.board.reset_board()
            self.ui.endScreen()
        print(w1, w2)
    
    def reset_game_root(self):
        self.game_root = ggNode(None, game_root.num_actions, game_root.state, game_root.cols, 0, self.board.dimensions)
    
    def get_ucbs_avg(self, ucbs):
        ucb_avg = 0
        count = 0
        for ucb in ucbs:
            if ucb != None:
                ucb_avg += ucb
                count += 1
        ucb_avg = ucb_avg/count
        return ucb_avg
    
    def get_cmd_line(self):
        cmd = ""
        root_val = -1
        node = None
        while cmd!="exit":
            cmd = input(">> ")
            cmd = cmd.strip()
            flag = False
            if " " in cmd:
                cmd = cmd.split(" ")
                flag = True
            if cmd == "p1":
                root_val = 1
                node = self.game_root
            elif cmd == "p2":
                root_val = 2
                node = self.game_root
            elif cmd == "show":
                if node == None:
                    print("No node connected.")
                else:
                    node.show_params()
            elif flag and cmd[0] == "down": 
                if node != None:
                    index = int(cmd[1])
                    if len(node.children) > 0:
                        node = node.children[index]
            elif cmd == "up":
                if node != None:
                    node = node.go_up()
            elif cmd == "exit":
                break
            else:
                print("Wrong command.")
            
board = ggBoard((6, 7), 4, True)
state =  board.board.copy()
cols = board.cols.copy()
num_actions = -1
if board.is_drop:
    num_actions = board.y
else:
    num_actions = board.max_moves
game_root = ggNode(None, num_actions, state, cols, board.move_count, board.dimensions)
game = GGP(game_root, board, "Connect-4")
game.play_train(20, 300, 600)
game.play_against_p1(4, 7000)
game.play_against_p2(4, 7000)

