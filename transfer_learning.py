# -*- coding: utf-8 -*-
"""
The transfer learning part is implemented by zhx
"""
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import game
from mcts_tr import SelfPlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GomokuNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(GomokuNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            512, 512, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(
            512 * (self.board_x - 4) * (self.board_y - 4), 1024
        )
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(
            -1, 1, self.board_x, self.board_y
        )  # batch_size x 1 x board_x x board_y
        s = F.relu(
            self.bn1(self.conv1(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn2(self.conv2(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn3(self.conv3(s))
        )  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(
            self.bn4(self.conv4(s))
        )  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, 512 * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=0.1,
            training=self.training,
        )  # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=0.1,
            training=self.training,
        )  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNetWrapper:
    def __init__(self, game, args):
        self.nnet = GomokuNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        if args.cuda:
            self.nnet.cuda()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.max_lr)

        # 1cycle learning rate parameters
        self.total_steps = args.numIters * args.epochs * (200000 // args.batch_size)
        self.current_step = 0

    def get_learning_rate(self):
        """Implement 1cycle learning rate strategy"""
        if self.current_step >= self.total_steps:
            return self.args.min_lr

        # Divide the total steps into two phases
        half_cycle = self.total_steps // 2

        if self.current_step <= half_cycle:
            # First phase: increase from min_lr to max_lr
            phase = self.current_step / half_cycle
            lr = self.args.min_lr + (self.args.max_lr - self.args.min_lr) * phase
        else:
            # Second phase: decrease from max_lr to min_lr
            phase = (self.current_step - half_cycle) / half_cycle
            lr = self.args.max_lr - (self.args.max_lr - self.args.min_lr) * phase

        return lr

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                # Update learning rate
                lr = self.get_learning_rate()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.current_step += 1

                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                if self.args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, lr=f"{lr:.1e}")

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()

                # Add gradient clipping
                if self.args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.args.grad_clip)

                self.optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float32))
        if self.args.cuda:
            board = board.cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        folder = folder.rstrip('/')
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint["state_dict"])


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def transfer_model(pretrained_model, new_game, new_args, freeze_layers=True):
    """
    Transfer learning function to adapt a pretrained model to a new board size.
    Optionally freeze layers for subsequent fine-tuning.

    Args:
        pretrained_model: The pretrained model on the 9x9 board.
        new_game: The game object for the new board (e.g., 15x15).
        new_args: Arguments for the new model (e.g., num_channels, dropout).
        freeze_layers: Whether to freeze convolutional layers initially.

    Returns:
        new_model: A new model adapted to the new board size.
    """
    new_model = GomokuNNet(new_game, new_args)  # Initialize new model

    # Transfer convolutional layers
    new_model.conv1.weight.data = pretrained_model.conv1.weight.data.clone()
    new_model.conv1.bias.data = pretrained_model.conv1.bias.data.clone()
    new_model.conv2.weight.data = pretrained_model.conv2.weight.data.clone()
    new_model.conv2.bias.data = pretrained_model.conv2.bias.data.clone()
    new_model.conv3.weight.data = pretrained_model.conv3.weight.data.clone()
    new_model.conv3.bias.data = pretrained_model.conv3.bias.data.clone()
    new_model.conv4.weight.data = pretrained_model.conv4.weight.data.clone()
    new_model.conv4.bias.data = pretrained_model.conv4.bias.data.clone()

    # Optionally freeze convolutional layers
    if freeze_layers:
        for param in [
            *new_model.conv1.parameters(),
            *new_model.conv2.parameters(),
            *new_model.conv3.parameters(),
            *new_model.conv4.parameters(),
        ]:
            param.requires_grad = False

    # Transfer batch normalization layers
    new_model.bn1.weight.data = pretrained_model.bn1.weight.data.clone()
    new_model.bn1.bias.data = pretrained_model.bn1.bias.data.clone()
    new_model.bn2.weight.data = pretrained_model.bn2.weight.data.clone()
    new_model.bn2.bias.data = pretrained_model.bn2.bias.data.clone()
    new_model.bn3.weight.data = pretrained_model.bn3.weight.data.clone()
    new_model.bn3.bias.data = pretrained_model.bn3.bias.data.clone()
    new_model.bn4.weight.data = pretrained_model.bn4.weight.data.clone()
    new_model.bn4.bias.data = pretrained_model.bn4.bias.data.clone()

    # Handle fully connected layers
    fc1_pretrained_size = pretrained_model.fc1.weight.data.size(1)
    fc1_new_size = new_model.fc1.weight.data.size(1)

    if fc1_pretrained_size <= fc1_new_size:
        new_model.fc1.weight.data[:, :fc1_pretrained_size] = pretrained_model.fc1.weight.data.clone()
        new_model.fc1.bias.data = pretrained_model.fc1.bias.data.clone()
    else:
        new_model.fc1.weight.data = pretrained_model.fc1.weight.data[:, :fc1_new_size].clone()
        new_model.fc1.bias.data = pretrained_model.fc1.bias.data.clone()

    # Fully reinitialize fc2, fc3, and fc4
    nn.init.xavier_uniform_(new_model.fc2.weight)
    nn.init.zeros_(new_model.fc2.bias)
    nn.init.xavier_uniform_(new_model.fc3.weight)
    nn.init.zeros_(new_model.fc3.bias)
    nn.init.xavier_uniform_(new_model.fc4.weight)
    nn.init.zeros_(new_model.fc4.bias)

    return new_model


args = dotdict({})
args.board_size = 15
# Training params
args.epochs = 10
args.batch_size = 256
args.numIters = 200
args.updateThreshold = 0.55
args.arenaCompare = 40
args.tempThreshold = 15

# Network params
args.min_lr = 1.0e-4
args.max_lr = 1.0e-2
args.grad_clip = 1.0

# MCTS params
args.numMCTSSims = 800
args.cpuct = 4.0

# System params
args.cuda = torch.cuda.is_available()
# args.cuda = False
args.checkpoint = "./temp"

# Step 1: Load 9x9 pretrained model
log.info("Loading 9x9 Game and Model...")
g_small = game.GomokuGame(9)
nnet_small = NNetWrapper(g_small, args)
nnet_small.load_checkpoint("./temp", "best.pth.tar")

# Step 2: Initialize 15x15 Game and Model
log.info("Initializing 15x15 Game and Model...")
g_large = game.GomokuGame(args.board_size)
nnet_large = NNetWrapper(g_large, args)

# Step 3: Transfer 9x9 model to 15x15 model
log.info("Transferring weights from 9x9 model to 15x15 model...")
nnet_large.nnet = transfer_model(nnet_small.nnet, g_large, args)

# Step 4: Save the transferred model (optional)
nnet_large.save_checkpoint("./temp", "transferred_15x15.pth.tar")

# Step 5: Start SelfPlay with the transferred model and start training
log.info("Loading the SelfCoach...")
s = SelfPlay(g_large, nnet_large, args)

log.info("Starting the learning process")
s.learn()
