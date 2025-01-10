import logging
import math
import numpy as np
from tqdm import tqdm
from collections import deque
from random import shuffle
import game


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: Since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] is not None:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][
                    a
                ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = -self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                    self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v


class SelfPlay:
    """
    This class executes the self-play + learning.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action
            )

            r = self.game.getGameEnded(board, self.curPlayer)

            if r is not None:
                # r * (1 if self.curPlayer == x[1] else -1) means 1 for winner, -1 for loser, 0 for draw.
                return [
                    (x[0], x[2], r * (1 if self.curPlayer == x[1] else -1))
                    for x in trainExamples
                ]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=200000)

            for _ in tqdm(range(100), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                    len(self.trainExamplesHistory) > 20
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = game.Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                    pwins + nwins == 0
                    or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )
