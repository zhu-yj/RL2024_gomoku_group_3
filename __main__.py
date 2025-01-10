import json
import torch
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet 
# author: 朱永健
# botzone 每次会运行main，没有保存历史的，所以要将历史输入到board中
# 但是如果等待网络输出做出回应，可能随着时间延长会变慢，所以直接用do_move更新board了，如果没有用到历史的state可能是可行的，但是要用到start_play，其实可以参考下俊康的处理

class BotzoneHuman(object):
    """
    Botzone human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        print("Error! Current player should be AI!!!")
        move = -100 # one error move
        # try:
        #     location = [all_requests[-1]["x"], all_requests[-1]["y"]] # 获取botzone上人类最近的输入
        #     # 判断是否有新输入
        #     if location != his_location:
        #         move = board.location_to_move(location) # 是人类新的输入
        #         his_location = location
        # except Exception as e:
        #     move = -1
        # if move == -1 or move not in board.availables:
        #     print("invalid move")
        #     move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)
    
def main():
    n = 5
    width, height = 15, 15
    model_file = './data/best_policy_15_15_5.model'
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)


    best_policy = PolicyValueNet(width, height, model_file = model_file,use_gpu=False)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                c_puct=5,
                                n_playout=10)  # set larger n_playout for better performance


    fullInput = json.loads(input())
    all_requests = fullInput["requests"] # 人类在botzone上下的棋子
    all_responses = fullInput["responses"] # AI下的棋

    # Botzone human player
    human = BotzoneHuman()

    # set start_player=0 for human first, 1 for AI
    if all_requests[0]['x'] == -1: # AI 先手
        start_player = 1
    else:
        start_player = 0
    x,y = game.start_botzone_play(player1=human, player2=mcts_player, start_player=start_player, is_shown=False, all_requests=all_requests, all_responses=all_responses)
    # print(x,y)
    print(json.dumps({"response": {"x":int(x), "y":int(y)}}))



if __name__ == '__main__':
  main()

