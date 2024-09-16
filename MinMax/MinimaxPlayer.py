from MinMax import BasicPlayers
import time

class MinimaxPlayer(BasicPlayers.RandomPlayer):
    """
    实现了最小最大算法的玩家。继承自随机玩家以进行随机移动。
    使用最小最大算法结合α-β剪枝提高速度，并通过迭代加深来满足时间限制。
    """
    def __init__(self, playerIndex, colour="red", timeLimit=0.5, maxDepth=20):
        """
        针对最小最大玩家的重写，加入了时间限制和最大搜索深度。
        这是给玩家选择棋步的时间限制，单位为秒。
        参数:
            playerIndex(int): 游戏中的玩家索引
            colour(str): UI渲染用的颜色。默认是红色
            timeLimit(int/float): 选择棋步的时间限制，单位为秒
            maxDepth(int): 计算机玩家可以达到的最大深度
        """
        self.index = playerIndex
        self.colour = colour
        self.timeLimit = timeLimit
        self.maxDepth = maxDepth

    def chooseMove(self, game):
        """
        为最小最大玩家选择棋步。这是搜索的开始，检查当前玩家可以做的所有移动。实现了迭代加深。
        当时间限制到达时，选择当前最优得分的棋步。
        参数:
            game(Game): 玩家正在进行移动的游戏
        返回:
            3元组[int]: 要做的棋步
        """
        moves = game.get_all_legal_moves()
        bestMove = (0, 0, 0)
        bestScore = -10000
        currentMaxDepth = 1
        startTime = time.time()
        # 初始时搜索深度为1，然后逐步增加深度。
        # 继续增加深度，直到时间限制到达，然后返回当前找到的最佳棋步。这是迭代加深。
        while time.time() - startTime <= self.timeLimit and currentMaxDepth <= self.maxDepth:
            for move in moves:
                # 模拟这个棋步并找到其得分
                copyGame = self.makeMove(game, move)
                score = self.getScore(copyGame, currentMaxDepth, -10000, 10000)
                # 返回得分最高的棋步
                if score >= bestScore:
                    bestScore = score
                    bestMove = move
                # 如果达到时间限制，则退出循环
                if time.time() - startTime >= self.timeLimit:
                    break
            # 增加当前最大搜索深度以进行迭代加深
            currentMaxDepth += 1

        # 万一选择了一个无效的棋步，或没有选择棋步。
        if game.is_legal_move(bestMove):
            return bestMove
        else:
            return self.randomMove(game)

    def getScore(self, game, depth, alpha, beta):
        """
        最小最大算法的递归部分。实现了α-β剪枝。
        递归地搜索树的底部以找到可能的得分。
        参数:
            game(Game): 从哪个游戏状态分支
            depth(int): 当前搜索的深度
            alpha(int): α值
            beta(int): β值
        返回:
            int
        """
        # 当到达树的底部时，返回静态评估
        if depth <= 0 or game.is_finished():
            return self.evaluate(game)

        moves = game.get_all_legal_moves()
        # 记录当前玩家
        currentPlayer = game.currentPlayer
        # 根据轮到哪个玩家设置bestScore的高低值，并设置maximise标志
        if currentPlayer == self.index:
            bestScore = -10000
            maximise = True
        else:
            bestScore = 10000
            maximise = False
        for move in moves:
            # 执行这个棋步并获取游戏的下一个状态
            copyGame = self.makeMove(game, move)
            # 递归调用
            score = self.getScore(copyGame, depth-1, alpha, beta)
            # 根据是最小化节点还是最大化节点进行不同的操作
            if maximise:
                bestScore = max(score, bestScore)
                alpha = max(alpha, bestScore)
            else:
                bestScore = min(score, bestScore)
                beta = min(beta, bestScore)
            # α-β剪枝
            if beta <= alpha:
                break
        return bestScore

    def evaluate(self, game):
        """
        评估特定的游戏状态。获取静态得分。
        参数:
            game(Game): 要评估的游戏状态
        返回:
            int: 从游戏状态计算的得分
        """
        # 找到自己的索引和其他玩家的索引
        if self.index == 1:
            otherIndex = 2
        else:
            otherIndex = 1
        score = 0
        scores = game.get_scores()
        # 每个玩家拥有的盒子增加10分
        score += 10*scores[self.index]
        # 每个对手拥有的盒子减少10分
        score -= 10*scores[otherIndex]
        # 评估需要根据轮到谁来调整
        # 如果轮到我们下棋，则希望完成盒子
        if game.currentPlayer == self.index:
            for i in range(game.height-1):
                for j in range(game.width-1):
                    no_sides = game.boxes[i][j].sides_completed()
                    # 0或1，没关系
                    if no_sides == 0:
                        pass
                    elif no_sides == 1:
                        score += 1
                    # 只有两个
                    elif no_sides == 2:
                        score -= 2
                    # 三个边
                    elif no_sides == 3:
                        score += 5

        # 如果轮到对手下棋，则不希望对手占领box
        elif game.currentPlayer == otherIndex:
            for i in range(game.height-1):
                for j in range(game.width-1):
                    no_sides = game.boxes[i][j].sides_completed()
                    # 0或1，没关系
                    if no_sides == 0 or no_sides == 1:
                        pass
                    # 只有两个，我们希望对手完成第三个
                    elif no_sides == 2:
                        score += 1
                    # 三个边意味着对手可以完成第四个并获得分数
                    elif no_sides == 3:
                        score -= 5
        return score

    def makeMove(self, game, move):
        """
        接受一个游戏状态和一个棋步，复制游戏并在其中执行这个棋步。
        参数:
            game(Game): 要进行的游戏
            move(Tuple[int]): 要执行的棋步
        返回:
            Game: 执行棋步后的游戏深拷贝
        """
        copyGame = game.get_copy()
        copyGame.take_turn(move)
        return copyGame

    def __str__(self):

        return "{}_minimax".format(self.index)
