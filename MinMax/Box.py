from .Line import Line


class Box:
    def __init__(self, top, bottom, left, right, owner=0):
        """
        Initialise a new box with its four edges and owner.
        Args:
            top, bottom, left, right: Line
        """
        self.completed = False
        self.owner = owner
        self.edges = [top, bottom, left, right]
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def sides_completed(self):
        """
        Function that returns how many sides of the 4 have been completed.
        Returns:
            int
        """
        return sum([bool(x) for x in self.edges])

    def check_completed(self, player):
        """
        Checks all owned edges. If all edges are owned, then this box is now owned
        by player.
        Args:
            player: int
        Returns:
            bool - only returns True if the box was completed on this turn.
                not if it was previously completed.
        """
        if not self.completed:
            if all(self.edges):
                #print("Player {} got a box!".format(player))
                self.owner = player
                self.completed = True
                return True
        return False

    def __str__(self):
        """
        String representation for Box. If there is no owner, return " ". Otherwise,
        return player number as string.
        Returns:
            str
        """
        if self.owner == 0:
            return " "
        else:
            return str(self.owner)

    def __bool__(self):
        """
        Define truth value for Box. If the Box is owned, Box is True.
        """
        return self.owner != 0
