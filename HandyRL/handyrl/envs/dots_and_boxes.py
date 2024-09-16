import torch
from torch import nn
import torch.nn.functional as F
from HandyRL.handyrl.environment import BaseEnvironment
import numpy as np


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(
            filters0, filters1, kernel_size,
            stride=1, padding=kernel_size // 2, bias=bias
        )
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):
    def __init__(self, input_size, out_filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters

        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h


class SimpleConv2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 3, 32

        self.conv = nn.Conv2d(1, filters, 3, stride=1, padding=1)  # Set input channels to 1
        self.blocks = nn.ModuleList([Conv(filters, filters, 3, bn=True) for _ in range(layers)])
        self.head_p = Head((filters, 11, 6), 2, 66)
        self.head_v = Head((filters, 11, 6), 1, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = F.relu(block(h))
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return {'policy': h_p, 'value': torch.tanh(h_v)}


class Environment(BaseEnvironment):
    def __init__(self, args=None):
        super().__init__()
        self.dots = 6
        self.box_state = np.zeros(shape=[5, 5], dtype=int)  # record boxes state only
        self.board_state = np.zeros(shape=[11, 6], dtype=np.float32)
        self.board_state[::2, -1] = -1

        self.player = -1  # player1: 1; player2: -1

        # flag of player1: 255, player2: 253, used to mark boxes;
        # flag = self.player_flag[self.player] ->255 if self.player is 1 else 253
        self.player_flag = [None, 255, 253]
        self.switch = True  # switch player or not

        self.done = False  # game over or not

        self.winner = 0

        self.mark_num = 0
        self.chess_num = []

    # reset the env per episode(epoch)
    def reset(self, args=None):
        #  clear the chessboard
        self.box_state = np.zeros(shape=[5, 5], dtype=int)
        self.board_state = np.zeros(shape=[11, 6], dtype=np.float32)
        self.board_state[::2, -1] = -1
        self.player = -1
        self.switch = True
        self.done = False

        self.winner = 0

        self.mark_num = 0
        self.chess_num = []

    def players(self):
        return [1, -1]

    def turn(self):
        if self.switch:
            self.player = - self.player

        return self.player

    def outcome(self):
        if not self.done:
            return {1: 0, -1: 0}

        self.get_winner()
        return {1: self.winner, -1: -self.winner}

    def reward(self):
        r = self.mark_num * 0.5

        for chess in self.chess_num:
            if chess == 2:
                r += 0.5

            elif chess == 1:
                r += 0.2

        # _r = np.exp(-(2 + abs(5 - action[0])))
        # _r += np.exp(-(1.5 + abs(2.5 - action[1])))
        # r += _r if _r >= 0.0195 else 0

        return {self.player: r, -self.player: 0}

    def play(self, action, _=None):
        flag = self.player_flag[self.player]
        self.switch = True
        row, col = self.num2site(action)
        # print(f'Player: {self.player}')
        # print(f'Action: {row, col}')

        self.board_state[row][col] = 1

        self.chess_num, boxes = self.is_surrounded(row, col)
        self.mark_num = len(boxes)

        if boxes:
            for box in boxes:
                self.mark_box(box=box, flag=flag)
            # surrounded boxes aren't None, means current player occupied boxes, no need to change player
            self.switch = False

    # judge if a box is surrounded after a certain action('*action'), may have a better method
    def is_surrounded(self, *action):
        boxes = []
        chess_num = []
        row, col = action[0], action[1]

        if row % 2 == 0:  # parallel chess
            if row not in (0, 10):  # not edge chess, could affect 2 boxes
                chess1 = self.board_state[row - 2][col] + self.board_state[row - 1][col] + self.board_state[row - 1][
                    col + 1] + 1
                chess_num.append(chess1)
                chess2 = self.board_state[row + 2][col] + self.board_state[row + 1][col] + self.board_state[row + 1][
                    col + 1] + 1
                chess_num.append(chess2)
                if self.board_state[row - 2][col] == self.board_state[row - 1][col] == self.board_state[row - 1][
                    col + 1] == 1:
                    boxes.append([row // 2 - 1, col])
                if self.board_state[row + 2][col] == self.board_state[row + 1][col] == self.board_state[row + 1][
                    col + 1] == 1:
                    boxes.append([row // 2, col])
            elif row == 0:  # left chess, affect 1 box only
                chess = self.board_state[row + 2][col] + self.board_state[row + 1][col] + self.board_state[row + 1][
                    col + 1] + 1
                chess_num.append(chess)
                if self.board_state[row + 2][col] == self.board_state[row + 1][col] == self.board_state[row + 1][
                    col + 1] == 1:
                    boxes.append([row // 2, col])
            else:  # row == 10 right chess, affect 1 box only
                chess = self.board_state[row - 2][col] + self.board_state[row - 1][col] + self.board_state[row - 1][
                    col + 1] + 1
                chess_num.append(chess)
                if self.board_state[row - 2][col] == self.board_state[row - 1][col] == self.board_state[row - 1][
                    col + 1] == 1:
                    boxes.append([row // 2 - 1, col])
        else:  # vertical chess
            if col not in (0, 5):
                chess1 = self.board_state[row][col + 1] + self.board_state[row - 1][col] + self.board_state[row + 1][
                    col] + 1
                chess_num.append(chess1)
                chess2 = self.board_state[row][col - 1] + self.board_state[row - 1][col - 1] + \
                         self.board_state[row + 1][
                             col - 1] + 1
                chess_num.append(chess2)
                if self.board_state[row][col + 1] == self.board_state[row - 1][col] == self.board_state[row + 1][
                    col] == 1:
                    boxes.append([row // 2, col])
                if self.board_state[row][col - 1] == self.board_state[row - 1][col - 1] == self.board_state[row + 1][
                    col - 1] == 1:
                    boxes.append([row // 2, col - 1])
            elif col == 0:
                chess = self.board_state[row][col + 1] + self.board_state[row - 1][col] + self.board_state[row + 1][
                    col] + 1
                chess_num.append(chess)
                if self.board_state[row][col + 1] == self.board_state[row - 1][col] == self.board_state[row + 1][
                    col] == 1:
                    boxes.append([row // 2, col])
            else:  # col == 5
                chess = self.board_state[row][col - 1] + self.board_state[row - 1][col - 1] + self.board_state[row + 1][
                    col - 1] + 1
                chess_num.append(chess)
                if self.board_state[row][col - 1] == self.board_state[row - 1][col - 1] == self.board_state[row + 1][
                    col - 1] == 1:
                    boxes.append([row // 2, col - 1])

        return chess_num, boxes

    # mark surrounded box
    def mark_box(self, box, flag):
        x, y = box[0], box[1]
        self.box_state[x][y] = flag

    def terminal(self):
        self.done = True
        n1_boxes = np.argwhere(self.box_state == 255)
        n2_boxes = np.argwhere(self.box_state == 253)

        if len(n1_boxes) + len(n2_boxes) < 25:
            self.done = False

        return self.done

    def get_winner(self):

        n1_boxes = np.argwhere(self.box_state == 255)
        n2_boxes = np.argwhere(self.box_state == 253)

        self.winner = 1 if len(n1_boxes) > len(n2_boxes) else -1

    def observation(self, player=None):
        obs = self.board_state.reshape((1, 11, 6))
        return obs  # this array will be fed to neural network

    def net(self):
        return SimpleConv2dModel()

    def num2site(self, num):
        x = num // 6
        y = num % 6
        return x, y

    def site2num(self, site):
        return site[0] * 6 + site[1]

    def legal_actions(self, _=None):
        return [self.site2num([x, y]) for x, y in np.argwhere(self.board_state == 0)]
