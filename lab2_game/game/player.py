class Player:
    """A class that represents a player in a game"""

    def __init__(self, char: str, ai: bool, ai_depth: int = 0) -> None:
        """Initializes a player.

        Args:
            char: a single-character string to represent the player in textual representations of game state
            ai: boolean indicating if player is AI
            ai_depth: depth of the AI
        """
        if len(char) != 1:
            raise ValueError("Character that represents player should be of length 1")

        self.char = char
        self.ai = ai
        self.ai_depth = ai_depth
