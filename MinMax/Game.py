from MinMax.Box import Box
from MinMax.Line import Line


class Game:
    def __init__(self, width, height, maxPlayers=2, curPlayer=1, legalMoves=False, copy_grid=None, copy_boxes=None, movesMade=None):
        """
        Initialise the game with given width and height.
        If grid or boxes are passed, then create a copy of these objects.
        Args:
            width: int
            height: int
            maxPlayers: int (2)
            curPlayer: int (1)
            legalMoves: List(3-Tuple(int))
            copy_grid: List[][][Line]
            copy_boxes: List[][Box]
        """
        self.width = width
        self.height = height
        self.currentPlayer = curPlayer
        self.maxPlayers = maxPlayers
        self.legalMoves = legalMoves
        self.movesMade = movesMade
        if copy_grid is None and copy_boxes is None:
            self.build_game()
        else:
            self.build_from_copy(copy_grid, copy_boxes)

    def build_game(self):
        """
        Builds an empty game board with internal width and height.
        """
        # Build two lists of horizontal and vertical lines.
        self.grid = [
            [[Line() for j in range(self.width-1)] for i in range(self.height)],
            [[Line() for j in range(self.height-1)] for i in range(self.width)]
        ]
        # This means grid[0][0][0] is the top left horizontal line
        # grid[1][0][0] is the top left vertical line
        # Then build the box objects in a 2d list
        self.boxes = [[0 for i in range(self.width-1)] for j in range(self.height-1)]
        for i in range(self.height-1):
            for j in range(self.width-1):
                # Boxes are constructed with lines in the order [top, bottom, left, right]
                self.boxes[i][j] = Box(self.grid[0][i][j], self.grid[0][i+1][j], self.grid[1][j][i], self.grid[1][j+1][i])
        # Build a list of all legal moves that can be made
        self.get_all_legal_moves()
        self.movesMade = []

    def build_from_copy(self, copy_grid, copy_boxes):
        """
        Builds a game board from copied values.
        Args:
            copy_grid: List[][][Line]
            copy_boxes: List[][Box]
        """
        # Build two lists of horizontal and vertical lines using the input grid.
        self.grid = [
            [[Line(j.owner) for j in i] for i in copy_grid[0]],
            [[Line(j.owner) for j in i] for i in copy_grid[1]]
        ]
        # This means grid[0][0][0] is the top left horizontal line
        # grid[1][0][0] is the top left vertical line
        # Then build the box objects in a 2d list
        self.boxes = [[0 for i in range(self.width-1)] for j in range(self.height-1)]
        for i in range(self.height-1):
            for j in range(self.width-1):
                # Boxes are constructed with lines in the order [top, bottom, left, right]
                self.boxes[i][j] = Box(self.grid[0][i][j], self.grid[0][i+1][j], self.grid[1][j][i], self.grid[1][j+1][i], copy_boxes[i][j].owner)

    def get_copy(self):
        """
        Game returns a deep copy of itself.
        Returns:
            Game
        """
        return Game(
            self.width,
            self.height,
            self.maxPlayers,
            self.currentPlayer,
            self.legalMoves.copy(),
            self.grid,
            self.boxes,
            self.movesMade.copy())

    def increment_player(self):
        """
        Increments the player counter. Wraps around on self.maxPlayers.
        """
        self.currentPlayer += 1
        if self.currentPlayer > self.maxPlayers:
            self.currentPlayer = 1

    def take_turn(self, move):
        """
        Takes a turn for next player. Claims a line.
        Returns bool for success. True if player secures a box.
        Args:
            player: int
            move: 3-tuple(int)
        Returns:
            bool
        """
        if self.is_legal_move(move):
            move_results = []
            # Attempt to claim the line.
            self.grid[move[0]][move[1]][move[2]].draw(self.currentPlayer)
            # Take the move made out of the list of legal moves.
            self.legalMoves.remove(move)
            self.movesMade.append(move)
            #self.movesMade.append("{} - {}".format(self.currentPlayer, move))
            #print("Made move {}".format(move))
            # Check the boxes associated with the line claimed.
            if not self.check_boxes_for_line(move):
                # If no box has been claimed this round, increment the player counter
                # Otherwise, it is still this player's turn.
                self.increment_player()
        else:
            print("Illegal move {}".format(move))


    def check_boxes_for_line(self, move):
        """
        Takes an index for a Line and checks the boxes associated with that line
        for completion. Assigns boxes to player.
        Args:
            move: 3-tuple(int)
        Returns:
            bool: True if box is claimed, False if not.
        """
        # Make a list for the results as there could be 1 or 2.
        results = []
        if move[0] == 0:
            # If the Line is on an edge ([1 0 0], [0 0 0], etc) then it only has one
            # associated box. Check that box.
            if move[1] == 0:
                results.append(self.boxes[move[1]][move[2]].check_completed(self.currentPlayer))
            elif move[1] == self.height-1:
                results.append(self.boxes[move[1]-1][move[2]].check_completed(self.currentPlayer))
            # If the Line is not on the edge, it connects to two boxes. Check both.
            else:
                results.append(self.boxes[move[1]-1][move[2]].check_completed(self.currentPlayer))
                results.append(self.boxes[move[1]][move[2]].check_completed(self.currentPlayer))
        else:
            if move[1] == 0:
                results.append(self.boxes[move[2]][move[1]].check_completed(self.currentPlayer))
            elif move[1] == self.width-1:
                results.append(self.boxes[move[2]][move[1]-1].check_completed(self.currentPlayer))
            else:
                results.append(self.boxes[move[2]][move[1]-1].check_completed(self.currentPlayer))
                results.append(self.boxes[move[2]][move[1]].check_completed(self.currentPlayer))
        # If any boxes were claimed, any(results) will return true.
        return any(results)


    def is_legal_move(self, move):
        """
        Checks if a certain move is legal.
        Args:
            move: 3-tuple(int)
        Returns:
            bool
        """
        return move in self.legalMoves

    def get_all_legal_moves(self, generate=False):
        """
        Finds all legal moves that can be made. Returns these as a list.
        Args:
            generate(Bool): If True, force the game to make a new list of legal
                moves
        Returns:
            List[3-tuple(int)]
        """
        # When a new Game is instanciated, legalMoves will be False, so this
        # Will generate a list of all possible legal moves
        # Once this list has been created, it can be stored and moves that are made
        # can be removed from the list, preventing this costly move generation.
        if self.legalMoves is False or generate:
            self.legalMoves = []
            o = 0
            for i in range(self.height):
                for j in range(self.width-1):
                    if not self.grid[o][i][j]:
                        self.legalMoves.append((o, i, j))
            o = 1
            for i in range(self.width):
                for j in range(self.height-1):
                    if not self.grid[o][i][j]:
                        self.legalMoves.append((o, i, j))
        return self.legalMoves.copy()

    def is_finished(self):
        """
        Checks if the game is finished or if there are still moves to be made.
        Returns:
            bool
        """
        return len(self.legalMoves) == 0

    def finish_game(self):
        """
        Function called when the game is finished. Declares a winner.
        """
        scores = [self.check_score(x) for x in range(1, self.maxPlayers+1)]
        #print("Game Finished!")
        #print("The winner is player {}, with {} boxes!".format(scores.index(max(scores))+1, max(scores)))

    def get_scores(self):
        """
        returns the scores for all players that have any score.
        Also returns number of unclaimed boxes.
        Returns:
            dict{int:int}
        """
        scores = {0:0, 1:0, 2:0}
        for i in range(self.height-1):
            for j in range(self.width-1):
                owner = self.boxes[i][j].owner
                scores[owner] += 1
        return scores

    def check_score(self, player):
        """
        Get and return the score for one player.
        Args:
            player: int
        Returns:
            int
        """
        score = 0
        for i in range(self.height-1):
            for j in range(self.width-1):
                if self.boxes[i][j].owner == player:
                    score += 1
        return score

    def winner(self):
        """
        If the game is finished, find the winner.
        """
        if self.is_finished():
            scores = self.get_scores()
            if scores[1] == scores[2]:
                return 0
            elif scores[1] > scores[2]:
                return 1
            else:
                return 2
        #print("Game is not yet finished!")
        return 0

    def save_statistics(self, filename, mode="a+"):
        """
        Takes all relevant statistics from the game and saves them to given filename.
        Saves -
            size of board
            all moves made
            final score
        Args:
            filename: str
            mode: str
        """
        if mode not in ["a", "w", "a+", "w+"]:
            mode = "a+"
        scores = self.get_scores()
        scoresStr = "{}, {}".format(scores[1], scores[2])
        gameStr = "{}x{}".format(self.width, self.height)
        try:
            with open(filename, mode) as outfile:
                outfile.write(gameStr+"\n")
                for line in self.movesMade:
                    outfile.write(str(line)+"\n")
                outfile.write(scoresStr+"\n")
        except Exception as e:
            print("Saving to results file {} failed.".format(filename))
            #print(e)

    def print_grid(self):
        """
        Prints an ascii representation of the board.
        For horizontal edges, '- -' is unclaimed, '---' is claimed
        For vertical edges, '¦' is unclaimed, '|' is claimed
        An empty box is unclaimed. A box with a number in is claimed.
        """
        for i in range(self.height):
            for j in range(self.width):
                if j != self.width-1:
                    print("*", end="")
                    if self.grid[0][i][j]:
                        print("---", end="")
                    else:
                        print("- -", end="")
                else:
                    print("*")
            if i != self.height-1:
                for j in range(self.width):
                    if j != self.width-1:
                        if self.grid[1][j][i]:
                            print("|", end="")
                        else:
                            print("¦", end="")
                        print(" {} ".format(self.boxes[i][j]), end="")
                    else:
                        if self.grid[1][j][i]:
                            print("|")
                        else:
                            print("¦")

    def __eq__(self, other):
        """
        Builtin equality method for seeing if games are equal.
        Games are equal if they have the same dimensions, all of the lines
        are owned by the same players and it is the same players turn.
        """
        if self.width != other.width:
            return False

        if self.height != other.height:
            return False

        if self.currentPlayer != other.currentPlayer:
            return False

        o = 0
        for i in range(self.height):
            for j in range(self.width-1):
                if self.grid[o][i][j].owner == 0 and other.grid[o][i][j].owner != 0:
                    return False
                if self.grid[o][i][j].owner != 0 and other.grid[o][i][j].owner == 0:
                    return False
        o = 1
        for i in range(self.width):
            for j in range(self.height-1):
                if self.grid[o][i][j].owner == 0 and other.grid[o][i][j].owner != 0:
                    return False
                if self.grid[o][i][j].owner != 0 and other.grid[o][i][j].owner == 0:
                    return False

        for i in range(self.height-1):
            for j in range(self.width-1):
                if self.boxes[i][j].owner != other.boxes[i][j].owner:
                    return False


        return True
