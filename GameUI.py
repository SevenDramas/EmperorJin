import sys
from PyQt5.QtWidgets import (QWidget, QToolTip,
                             QPushButton, QApplication, QLabel, QSpinBox,
                             QComboBox, QColorDialog, QDoubleSpinBox, QRadioButton,
                             QMessageBox)
from PyQt5.QtGui import QFont, QIcon
from PyQt5 import QtCore
from MinMax.Game import Game
from MinMax import PlayerFactory


class StartFrame(QWidget):
    """
    Start frame class includes game options, and can launch the game with these options
    """

    def __init__(self):
        super().__init__()
        self.playerFactory = PlayerFactory.PlayerFactory()
        self.p1Col = 'orange'
        self.p2Col = 'lightBlue'
        self.initUI()

    def initUI(self):
        """
        Simple start window that asks for game options.
        Has a button that launches the game window.
        """

        self.setWindowTitle("点格棋")
        self.setWindowIcon(QIcon('images/logo.jpg'))
        self.setGeometry(50, 50, 600, 450)
        # Title label
        titleLabel = QLabel("点格棋", self)
        titleLabel.setFont(QFont('Arial', 20))
        titleLabel.resize(titleLabel.sizeHint())
        titleLabel.move(256, 25)

        # Input for size
        # Two number input boxes
        widthLabel = QLabel("宽度:", self)
        widthLabel.resize(widthLabel.sizeHint())
        widthLabel.move(180, 103)
        self.widthInput = QSpinBox(self)
        self.widthInput.resize(self.widthInput.sizeHint())
        self.widthInput.move(230, 100)
        self.widthInput.setRange(6, 10)
        self.widthInput.setToolTip("Select the width of the game board. This is the number of dots in each row.")

        heightLabel = QLabel("高度:", self)
        heightLabel.resize(heightLabel.sizeHint())
        heightLabel.move(330, 103)
        self.heightInput = QSpinBox(self)
        self.heightInput.resize(self.heightInput.sizeHint())
        self.heightInput.move(380, 100)
        self.heightInput.setRange(6, 10)
        self.heightInput.setToolTip("Select the height of the game board. This is the number of dots in each column.")

        # Input for players. Dropdowns filled with values from factory
        # Create label first
        playerOneLabel = QLabel("玩家一:", self)
        playerOneLabel.resize(playerOneLabel.sizeHint())
        playerOneLabel.move(170, 153)
        # Then create the dropdown and size it
        self.playerOneDropdown = QComboBox(self)
        self.playerOneDropdown.resize(125, 20)
        self.playerOneDropdown.move(240, 150)
        self.playerOneDropdown.currentIndexChanged.connect(self.playerPickerChanged)
        self.playerOneDropdown.setToolTip("Choose the player type for Player 1.")
        # Create the colour picker button for player colour
        self.playerOneColour = QPushButton("Colour", self)
        self.playerOneColour.resize(self.playerOneColour.sizeHint())
        self.playerOneColour.move(380, 145)
        self.playerOneColour.setStyleSheet("background-color: {}".format(self.p1Col))
        self.playerOneColour.clicked.connect(self.playerOneColourPicker)
        self.playerOneColour.setToolTip("Select the colour for Player 1s lines and boxes.")

        # These are the optional boxes that should only appear for specific player types.
        self.playerOneTimeLabel = QLabel("时间限制:", self)
        self.playerOneTimeLabel.resize(0, 0)
        self.playerOneTimeLabel.move(140, 188)

        self.playerOneTimeLimit = QDoubleSpinBox(self)
        self.playerOneTimeLimit.resize(0, 0)
        self.playerOneTimeLimit.move(210, 185)
        self.playerOneTimeLimit.setRange(0.2, 20.0)
        self.playerOneTimeLimit.setSingleStep(0.1)
        self.playerOneTimeLimit.setToolTip("Select the amount of time available to Player 1 to take their turn.")

        self.playerOneMaxDepthLabel = QLabel("最大深度:", self)
        self.playerOneMaxDepthLabel.resize(0, 0)
        self.playerOneMaxDepthLabel.move(400, 188)

        self.playerOneMaxDepth = QSpinBox(self)
        self.playerOneMaxDepth.resize(0, 0)
        self.playerOneMaxDepth.move(470, 185)
        self.playerOneMaxDepth.setRange(2, 50)
        self.playerOneMaxDepth.setValue(15)
        self.playerOneMaxDepth.setToolTip("Select the max search depth that Minimax can reach.")

        self.playerOneCValueLabel = QLabel("P1 C Value:", self)
        self.playerOneCValueLabel.resize(0, 0)
        self.playerOneCValueLabel.move(310, 178)

        self.playerOneCValue = QDoubleSpinBox(self)
        self.playerOneCValue.resize(0, 0)
        self.playerOneCValue.move(370, 175)
        self.playerOneCValue.setRange(0.0, 10.0)
        self.playerOneCValue.setSingleStep(0.05)
        self.playerOneCValue.setValue(1.4)
        self.playerOneCValue.setToolTip("Select the exploration coefficient for Monte Carlo Tree Search.")

        playerTwoLabel = QLabel("玩家二:", self)
        playerTwoLabel.resize(playerTwoLabel.sizeHint())
        playerTwoLabel.move(170, 233)
        self.playerTwoDropdown = QComboBox(self)
        self.playerTwoDropdown.resize(125, 20)
        self.playerTwoDropdown.move(240, 230)
        self.playerTwoDropdown.currentIndexChanged.connect(self.playerPickerChanged)
        self.playerTwoDropdown.setToolTip("Choose the player type for Player 2.")

        self.playerTwoColour = QPushButton("Colour", self)
        self.playerTwoColour.resize(self.playerTwoColour.sizeHint())
        self.playerTwoColour.move(380, 225)
        self.playerTwoColour.setStyleSheet("background-color: {}".format(self.p2Col))
        self.playerTwoColour.clicked.connect(self.playerTwoColourPicker)
        self.playerTwoColour.setToolTip("Select the colour for Player 2s lines and boxes.")

        self.playerTwoTimeLabel = QLabel("时间限制:", self)
        self.playerTwoTimeLabel.resize(0, 0)
        self.playerTwoTimeLabel.move(140, 268)

        self.playerTwoTimeLimit = QDoubleSpinBox(self)
        self.playerTwoTimeLimit.resize(0, 0)
        self.playerTwoTimeLimit.move(210, 265)
        self.playerTwoTimeLimit.setRange(0.2, 20.0)
        self.playerTwoTimeLimit.setSingleStep(0.1)
        self.playerTwoTimeLimit.setToolTip("Select the amount of time available to Player 2 to take their turn.")

        self.playerTwoMaxDepthLabel = QLabel("最大深度:", self)
        self.playerTwoMaxDepthLabel.resize(0, 0)
        self.playerTwoMaxDepthLabel.move(400, 268)

        self.playerTwoMaxDepth = QSpinBox(self)
        self.playerTwoMaxDepth.resize(0, 0)
        self.playerTwoMaxDepth.move(470, 265)
        self.playerTwoMaxDepth.setRange(2, 50)
        self.playerTwoMaxDepth.setValue(15)
        self.playerTwoMaxDepth.setToolTip("Select the max search depth that Minimax can reach.")

        self.playerTwoCValueLabel = QLabel("P2 C Value:", self)
        self.playerTwoCValueLabel.resize(0, 0)
        self.playerTwoCValueLabel.move(310, 258)

        self.playerTwoCValue = QDoubleSpinBox(self)
        self.playerTwoCValue.resize(0, 0)
        self.playerTwoCValue.move(370, 255)
        self.playerTwoCValue.setRange(0.0, 10.0)
        self.playerTwoCValue.setSingleStep(0.05)
        self.playerTwoCValue.setValue(1.4)
        self.playerTwoCValue.setToolTip("Select the exploration coefficient for Monte Carlo Tree Search.")

        # Populate both dropdowns with the player types in Player Factory.
        for player in self.playerFactory.playerTypes:
            self.playerOneDropdown.addItem(player)
            self.playerTwoDropdown.addItem(player)

        # Button to start game
        startButton = QPushButton('开始游戏', self)
        startButton.resize(100, 40)
        startButton.move(260, 350)
        startButton.clicked.connect(self.startGame)

        self.show()

    def playerPickerChanged(self):
        """
        Callback for when the value of a player type picker changes.
        """
        p1type = self.playerOneDropdown.currentText()
        p2type = self.playerTwoDropdown.currentText()

        if p1type == "Minimax Player":
            # If dropdown is minimax player, resize the minimax options to their correct sizes
            self.playerOneTimeLabel.resize(self.playerOneTimeLabel.sizeHint())
            self.playerOneTimeLimit.resize(50, 20)
            self.playerOneMaxDepthLabel.resize(self.playerOneMaxDepthLabel.sizeHint())
            self.playerOneMaxDepth.resize(40, 20)

            self.playerOneCValueLabel.resize(0, 0)
            self.playerOneCValue.resize(0, 0)

        else:
            # If it is neither of those, make all of these elements invisible.
            self.playerOneTimeLabel.resize(0, 0)
            self.playerOneTimeLimit.resize(0, 0)
            self.playerOneCValueLabel.resize(0, 0)
            self.playerOneCValue.resize(0, 0)
            self.playerOneMaxDepthLabel.resize(0, 0)
            self.playerOneMaxDepth.resize(0, 0)

        if p2type == "Minimax Player":
            # If dropdown is minimax player, resize the minimax options to their correct sizes
            self.playerTwoTimeLabel.resize(self.playerTwoTimeLabel.sizeHint())
            self.playerTwoTimeLimit.resize(50, 20)
            self.playerTwoMaxDepthLabel.resize(self.playerTwoMaxDepthLabel.sizeHint())
            self.playerTwoMaxDepth.resize(40, 20)

            self.playerTwoCValueLabel.resize(0, 0)
            self.playerTwoCValue.resize(0, 0)

        else:
            # If it is neither of those, make all of these elements invisible.
            self.playerTwoTimeLabel.resize(0, 0)
            self.playerTwoTimeLimit.resize(0, 0)
            self.playerTwoCValueLabel.resize(0, 0)
            self.playerTwoCValue.resize(0, 0)
            self.playerTwoMaxDepthLabel.resize(0, 0)
            self.playerTwoMaxDepth.resize(0, 0)

    def playerOneColourPicker(self):
        """
        Callback for p1 colour picker button
        """
        self.p1Col = QColorDialog.getColor().name()
        self.playerOneColour.setStyleSheet("background-color: {}".format(self.p1Col))

    def playerTwoColourPicker(self):
        """
        Callback for p2 colour picker button
        """
        self.p2Col = QColorDialog.getColor().name()
        self.playerTwoColour.setStyleSheet("background-color: {}".format(self.p2Col))

    def startGame(self):
        """
        Starts game. Creates two players with values from inputs using the Player Factory class.
        Then creates game and sends it options and players. Finally closes itself.
        """
        playerOne = self.playerFactory.makePlayer(
            self.playerOneDropdown.currentText(),
            1,
            self.p1Col,
            self.playerOneTimeLimit.value(),
            self.playerOneMaxDepth.value(),
            self.playerOneCValue.value()
        )

        playerTwo = self.playerFactory.makePlayer(
            self.playerTwoDropdown.currentText(),
            2,
            self.p2Col,
            self.playerTwoTimeLimit.value(),
            self.playerTwoMaxDepth.value(),
            self.playerTwoCValue.value()
        )

        players = [playerOne, playerTwo]

        game = Game(self.widthInput.value(), self.heightInput.value())
        self.gf = GameFrame(game, players)
        self.close()


class GameFrame(QWidget):

    def __init__(self, game, players, filename=False):
        """
        GameFrame is a QWidget that can also hold and read a Game instance, and a list
        of current players in the game.
        Args:
            width: int
            height: int
            players: List[PlayerBase]
        """
        super().__init__()
        self.players = players
        self.p1Colour = self.players[0].colour
        self.p2Colour = self.players[1].colour
        self.blockedColour = "Black"
        self.game = game
        self.width = game.width
        self.height = game.height
        self.resultsFilename = filename
        self.boxSize = 200
        self.lineWidth = 40
        self.initUI()

    def initUI(self):
        """
        Initialise the UI. Set up the window, turn and winner text labels.
        Set up the grid of lines as buttons that can be clicked, and set the boxes
        as labels with no value.
        """
        # work out window size before making
        winWidth = (self.boxSize + self.lineWidth) * (self.width - 1) + self.lineWidth + 20
        winHeight = 100 + (self.boxSize + self.lineWidth) * (self.height - 1) + self.lineWidth + 20
        # Set title
        self.setWindowTitle("点格棋")
        # self.setWindowIcon(QIcon('images/logo.jpg'))
        self.setGeometry(50, 50, winWidth, winHeight)

        # Set turn label
        self.titleLabel = QLabel("Player 1", self)
        self.titleLabel.setFont(QFont('Arial', 16))
        self.titleLabel.resize(self.titleLabel.sizeHint())
        self.titleLabel.move((winWidth // 2) - 40, 30)

        # Create winner label with no text
        self.winnerLabel = QLabel("", self)
        self.winnerLabel.setFont(QFont('Arial', 16))
        self.winnerLabel.move((winWidth // 2) - 65, 20)
        self.winnerLabel.resize(0, 0)

        # Create Play Again button but do not make visible.
        self.replayButton = QPushButton("Play Again!", self)
        self.replayButton.resize(2, 2)
        self.replayButton.move((winWidth // 2) - 50, 50)
        self.replayButton.clicked.connect(self.replay)

        # Build board.
        # Lists of buttons
        topLeftX = 10
        topLeftY = 100
        # Build an array of button objects - this is the same list constructor as
        # the one to build the array of Lines in Game.
        self.buttonGrid = [
            [[GameButton(self) for j in range(self.width - 1)] for i in range(self.height)],
            [[GameButton(self) for j in range(self.height - 1)] for i in range(self.width)]
        ]
        o = 0
        x = topLeftX
        y = topLeftY
        # Go through each button object and set its properties
        for i in range(self.height):
            x = topLeftX + self.lineWidth
            for j in range(self.width - 1):
                # create a 'dot'
                l = QLabel('', self)
                l.move(x - self.lineWidth, y)
                l.resize(self.lineWidth, self.lineWidth)
                l.setStyleSheet("border: 5px solid black; border-radius: 5px")
                # Make it the right size & orientation
                self.buttonGrid[o][i][j].resize(self.boxSize, self.lineWidth)
                # Place it
                self.buttonGrid[o][i][j].move(x, y)
                # Set the tooltip and value so we know which button is which
                self.buttonGrid[o][i][j].setToolTip('{} {} {}'.format(o, i, j))
                self.buttonGrid[o][i][j].value = (o, i, j)
                # Set the callback function
                self.buttonGrid[o][i][j].clicked.connect(self.lineClicked)
                # Disable the button so it can't be clicked when it shouldn't
                self.buttonGrid[o][i][j].setEnabled(False)
                # Adjust the offsets for next button
                x = x + self.lineWidth + self.boxSize
            # create an end dot
            l = QLabel('', self)
            l.move(x - self.lineWidth, y)
            l.resize(self.lineWidth, self.lineWidth)
            l.setStyleSheet("border: 5px solid black; border-radius: 5px")
            y = y + self.lineWidth + self.boxSize

        # now do it again for the verticals
        o = 1
        x = topLeftX
        for i in range(self.width):
            y = topLeftY + self.lineWidth
            for j in range(self.height - 1):
                self.buttonGrid[o][i][j].resize(self.lineWidth, self.boxSize)
                self.buttonGrid[o][i][j].move(x, y)
                self.buttonGrid[o][i][j].setToolTip('{} {} {}'.format(o, i, j))
                self.buttonGrid[o][i][j].value = (o, i, j)
                self.buttonGrid[o][i][j].clicked.connect(self.lineClicked)
                self.buttonGrid[o][i][j].setEnabled(False)
                y = y + self.lineWidth + self.boxSize
            x = x + self.lineWidth + self.boxSize

        # Now build grid of labels to show box owner.
        # Labels are initially empty strings, updated in update()
        self.boxes = [[QLabel("", self) for i in range(self.width - 1)] for j in range(self.height - 1)]
        y = topLeftY + self.lineWidth
        for i in range(self.height - 1):
            x = topLeftX + self.lineWidth
            for j in range(self.width - 1):
                self.boxes[i][j].resize(self.boxSize, self.boxSize)
                self.boxes[i][j].move(x, y)
                self.boxes[i][j].setAlignment(QtCore.Qt.AlignCenter)
                x = x + self.lineWidth + self.boxSize
            y = y + self.lineWidth + self.boxSize

        self.show()
        QApplication.processEvents()
        self.updateGame()
        # self.mainLoop()

    def mainLoop(self):
        """
        Main loop run every turn.
        Decides what happens when a human player or AI Player takes a turn.
        """
        currentPlayer = self.players[self.game.currentPlayer - 1]
        if currentPlayer.isHuman():
            self.humanTurn()
        else:
            # Send the AI player a copy of the game board right now and it will
            # return a move.
            move = currentPlayer.chooseMove(self.game.get_copy())
            self.makeMove(move)

    def humanTurn(self):
        """
        Logic that the game runs through when a human player is taking their turn.
        Enables all of the buttons that should be enabled.
        """
        # Enable all of the buttons that correspond to legal moves.
        for move in self.game.get_all_legal_moves():
            self.buttonGrid[move[0]][move[1]][move[2]].setEnabled(True)

    def updateGame(self):
        """
        Updates the display after each player's turn.
        """
        # Update turn label
        self.titleLabel.setText("Player {}".format(self.game.currentPlayer))
        self.titleLabel.resize(self.titleLabel.sizeHint())
        # update line colours
        for i in range(self.height):
            for j in range(self.width - 1):
                line_owner = self.game.grid[0][i][j].owner
                if line_owner == 1:
                    self.buttonGrid[0][i][j].setStyleSheet("background-color: {}".format(self.p1Colour))
                elif line_owner == 2:
                    self.buttonGrid[0][i][j].setStyleSheet("background-color: {}".format(self.p2Colour))
                elif line_owner == 3:
                    self.buttonGrid[0][i][j].setStyleSheet("background-color: {}".format(self.blockedColour))
        for i in range(self.width):
            for j in range(self.height - 1):
                line_owner = self.game.grid[1][i][j].owner
                if line_owner == 1:
                    self.buttonGrid[1][i][j].setStyleSheet("background-color: {}".format(self.p1Colour))
                elif line_owner == 2:
                    self.buttonGrid[1][i][j].setStyleSheet("background-color: {}".format(self.p2Colour))
                elif line_owner == 3:
                    self.buttonGrid[1][i][j].setStyleSheet("background-color: {}".format(self.blockedColour))
        # Update grid for box numbers
        for i in range(self.height - 1):
            for j in range(self.width - 1):
                owner = self.game.boxes[i][j].owner
                if owner != 0:
                    # This sets the text for owner number and sets box colour.
                    self.boxes[i][j].setText("{}".format(owner))
                    self.boxes[i][j].setStyleSheet("background-color: {}".format(self.players[owner - 1].colour))
        # Force the GUI to update. This is for games with no human player.
        QApplication.processEvents()
        # If the game isn't done yet, go back to the main loop.
        if not self.game.is_finished():
            self.mainLoop()
        # If the game is finished, set the winner label
        else:
            # Hide title label
            self.titleLabel.resize(0, 0)
            # Update the string that says who won.
            winner = self.game.winner()
            if winner == 0:
                winnerStr = "It's a draw!"
            else:
                winnerStr = "Player {} wins!".format(self.game.winner())
            print(winnerStr)
            self.winnerLabel.setText(winnerStr)
            self.winnerLabel.resize(self.winnerLabel.sizeHint())
            # self.winnerLabel.move(200, 50)
            self.replayButton.resize(100, 40)
            # self.replayButton.resize(self.replayButton.sizeHint())
            # If a results filename has been passed to GameFrame then save the
            # game stats to this location and close the frame.
            if self.resultsFilename:
                self.game.save_statistics(self.resultsFilename, "a+")
                self.close()

    def lineClicked(self):
        """
        Callback for 'line' buttons. Identifies which button was pressed, and then
        takes the turn for that button.
        """
        sender = self.sender()
        self.disableAllButtons()
        self.makeMove(sender.value)

    def makeMove(self, move):
        """
        Makes a move.
        Args:
            move: 3-tuple[int]
        """
        print("Player {} making move {}.".format(self.game.currentPlayer, move))
        # Set the colour of the line that was just played.
        button = self.buttonGrid[move[0]][move[1]][move[2]]
        if self.game.currentPlayer == 1:
            button.setStyleSheet("background-color: {}".format(self.p1Colour))
        elif self.game.currentPlayer == 2:
            button.setStyleSheet("background-color: {}".format(self.p2Colour))
        # Then send the move to the game
        self.game.take_turn(move)
        self.updateGame()

    def disableAllButtons(self):
        """
        Disables all buttons that are part of the game grid.
        """
        o = 0
        for i in range(self.height):
            for j in range(self.width - 1):
                self.buttonGrid[o][i][j].setEnabled(False)
        o = 1
        for i in range(self.width):
            for j in range(self.height - 1):
                self.buttonGrid[o][i][j].setEnabled(False)

    def replay(self):
        """
        Replay function. Creates a new start frame and then destroys itself.
        """
        self.sf = StartFrame()
        self.close()


class GameButton(QPushButton):
    """
    A very simple subclassed version of QPushButton that also holds a value.
    This is so when a button is pressed in the game, it knows its corresponding move
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.value = (-1, 0, 0)


def main():
    app = QApplication(sys.argv)
    ex = StartFrame()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
