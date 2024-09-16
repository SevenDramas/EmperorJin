import time

class Line:
    def __init__(self, owner=0):
        """
        Initialise new Line object.
        Args:
            owner: int
        """
        self.owner = owner

    def draw(self, player):
        """
        'draw' a line for a certain player. That player then owns the line.
        Args:
            player: int
        Returns:
            bool: True if line is successfully claimed.
        """
        if self.owner == 0:
            self.owner = player
            return True
        return False

    def __bool__(self):
        """
        Define truth value for line. If the line is owned, Line is True.
        """
        return self.owner != 0

