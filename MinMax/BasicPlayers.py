import random
import time

class PlayerBase:
    def __init__(self, playerIndex, colour="red"):
        self.index = playerIndex
        self.colour = colour

    def isHuman(self):
        pass

    def __str__(self):
        return "{}_player".format(self.index)

class HumanPlayer(PlayerBase):
    def __init__(self, playerIndex, colour="blue"):
        super().__init__(playerIndex, colour)

    def isHuman(self):

        return True

    def __str__(self):

        return "{}_human".format(self.index)

class RandomPlayer(PlayerBase):

    def chooseMove(self, game):

        return self.randomMove(game)

    def randomMove(self, game):

        #time.sleep(0.25)
        return random.choice(game.get_all_legal_moves())

    def isHuman(self):

        return False

    def __str__(self):

        return "{}_random".format(self.index)

class MovesInOrder(PlayerBase):

    def chooseMove(self, game):
        moves = game.get_all_legal_moves()
        # 从列表中间选择一个移动
        ind = int(len(moves)*0.5)
        return moves[ind]

    def __str__(self):

        return "{}_ordered".format(self.index)
