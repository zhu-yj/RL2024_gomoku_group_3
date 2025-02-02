# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        self.history_states = deque(maxlen=3)
        for _ in range(3):
            self.history_states.append(np.zeros([4,self.width,self.width]))
        self.temp_states = np.zeros([10,self.width,self.width])
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2


    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.history_states = deque(maxlen=3)
        for _ in range(3):
            self.history_states.append(np.zeros([4,self.width,self.width]))
        self.temp_states = np.zeros([10,self.width,self.width])
        self.last_move = -1


    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        # 陈俊康修改，调整了特征平面的输入（加入历史3步）
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        # print("a")
        p1, p2 = self.board.players
        # print("b")
        player1.set_player_ind(p1)
        # print("c")
        player2.set_player_ind(p2)
        # print("d")
        players = {p1: player1, p2: player2}
        # print("e")
        if is_shown:
            # print("f")
            self.graphic(self.board, player1.player, player2.player)
        while True:
            # print("g")
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            print("self.board.temp_states[0]", self.board.temp_states[0])
            current_state = self.board.current_state()
            temp_states = []
            for his_state in self.board.history_states:
                temp_states.append(his_state[:2])
                # print("his_state", his_state.shape)
            temp_states.append(current_state)
            # print("current_states", current_state.shape)
            temp_states = np.concatenate(temp_states, axis=0)
            # print("temp_states.shape:", temp_states.shape)  
            self.board.temp_states = temp_states
            self.board.history_states.append(current_state)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                    
                return winner
            
    def start_botzone_play(self, player1, player2, all_requests, all_responses, start_player=0, is_shown=1):
        # 朱永健，对应botzone交互
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        
        if len(all_responses) == 0 and all_requests[0]["x"] == -1: # 下第一个棋子，且AI先手
            current_player = self.board.get_current_player() # current_player是AI
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            [x,y]=self.board.move_to_location(move) # 给botzone
            return x,y
        
        if all_requests[0]["x"] == -1: # AI 先手
            for i in range(len(all_requests)-1):
                move = self.board.location_to_move([all_responses[i]["x"],all_responses[i]["y"]])
                self.board.do_move(move)
                move = self.board.location_to_move([all_requests[i+1]["x"],all_requests[i+1]["y"]])
                self.board.do_move(move)
        else:
            for i in range(len(all_requests)-1): # 人类先手
                move = self.board.location_to_move([all_requests[i]["x"],all_requests[i]["y"]])
                self.board.do_move(move)
                move = self.board.location_to_move([all_responses[i]["x"],all_responses[i]["y"]])
                self.board.do_move(move)
            move = self.board.location_to_move([all_requests[-1]["x"],all_requests[-1]["y"]])
            self.board.do_move(move)
        
        current_player = self.board.get_current_player() # current_player是AI
        player_in_turn = players[current_player]
        move = player_in_turn.get_action(self.board)
        [x,y]=self.board.move_to_location(move) # 给botzone
        return x,y
    
        # while True: # 应该不用while，botzone会自己判断胜负，只需要输出一个单步就行，甚至不用do_move，不需要更新棋盘状态，下次用
        #     current_player = self.board.get_current_player()
        #     player_in_turn = players[current_player]
        #     move = player_in_turn.get_action(self.board)
        #     # 将AI的move输出到botzone
        #     if player_in_turn == p2: # AI
        #         [x,y]=self.board.move_to_location(move)
        #         print(json.dumps({"response": {"x": x, "y": y}}))#输出到location
        #     self.board.do_move(move)
        #     if is_shown:
        #         self.graphic(self.board, player1.player, player2.player)
        #     end, winner = self.board.game_end()
        #     if end:
        #         if is_shown:
        #             if winner != -1:
        #                 print("Game end. Winner is", players[winner])
        #             else:
        #                 print("Game end. Tie")
        #         return winner
            

    def start_botzone_play2(self, player1, player2, start_player=0, is_shown=1):
        # 朱永健，对应陈俊康对增加特征的改进，实现改进后与botzone交互
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        # print("a")
        p1, p2 = self.board.players
        # print("b")
        player1.set_player_ind(p1)
        # print("c")
        player2.set_player_ind(p2)
        # print("d")
        players = {p1: player1, p2: player2}
        # print("e")
        if is_shown:
            # print("f")
            self.graphic(self.board, player1.player, player2.player)
        while True:
            # print("g")
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            print("self.board.temp_states[0]", self.board.temp_states[0])
            current_state = self.board.current_state()
            temp_states = []
            for his_state in self.board.history_states:
                temp_states.append(his_state[:2])
                # print("his_state", his_state.shape)
            temp_states.append(current_state)
            # print("current_states", current_state.shape)
            temp_states = np.concatenate(temp_states, axis=0)
            # print("temp_states.shape:", temp_states.shape)  
            self.board.temp_states = temp_states
            self.board.history_states.append(current_state)


            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner


    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        # 陈俊康修改，调整了特征平面的输入（加入历史3步）
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            current_state = self.board.current_state()
            temp_states = []
            for his_state in self.board.history_states:
                temp_states.append(his_state[:2])
                # print("his_state", his_state.shape)
            temp_states.append(current_state)
            # print("current_states", current_state.shape)
            temp_states = np.concatenate(temp_states, axis=0)
            # print("temp_states.shape:", temp_states.shape)  
            self.board.temp_states = temp_states
            self.board.history_states.append(current_state)
            states.append(self.board.temp_states)
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                # print("states:", len(states), states[0].shape)
                # print("mcts_probs:", len(mcts_probs), mcts_probs[0].shape)
                # print("winners_z:", winners_z.shape, winners_z)
                return winner, zip(states, mcts_probs, winners_z)
