from typing import Dict, Iterable, List, Optional, Tuple
import random
from lab2.game.game import Game
from lab2.game.move import Move
from lab2.game.player import Player
from lab2.game.state import State


class DotsAndBoxes(Game):
    """Class that represents the dots and boxes game"""
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, rows: int = 2, cols: int = 2, first_player: Player = None, second_player: Player = None, max_ai_depth: int = 3):
        """
        Initializes game.

        Parameters:
            rows, cols: the size of the game as number of columns and rows of boxes
            first_player: the player that will go first (if None is passed, a player will be created)
            second_player: the player that will go second (if None is passed, a player will be created)
        """
        self.first_player = first_player or Player(self.FIRST_PLAYER_DEFAULT_CHAR, True)
        self.second_player = second_player or Player(self.SECOND_PLAYER_DEFAULT_CHAR, True)

        state = DotsAndBoxesState(self.first_player, self.second_player, rows, cols, max_ai_depth)

        super().__init__(state)


class DotsAndBoxesMove(Move):
    """
    Class that represents a move in the dots and boxes game

    Variables:
        connection: str, 'h' if a horizontal line or 'v' if a vertical line
        loc: line coordinates as a tuple: (column, row) for horizontal or (row, column) for vertical
    """
    def __init__(self, connection: str, loc: Tuple[int, int]):
        self.connection = connection
        self.loc = loc

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DotsAndBoxesMove):
            return False
        return self.connection == o.connection and self.loc == o.loc


class DotsAndBoxesState(State):
    """Class that represents a state in the dots and boxes game"""
    def __init__(self, current_player: Player, other_player: Player, rows: int = 2, cols: int = 2, ai_depth: int = 3, horizontals: List[List[bool]] = None, verticals: List[List[bool]] = None, boxes: List[List[Player]] = None):
        """Creates the state. Do not call directly."""
        self.ai_depth = ai_depth

        if horizontals and verticals and boxes:
            self.rows = rows
            self.cols = cols
            self.horizontals = horizontals
            self.verticals = verticals
            self.boxes = boxes

        elif rows is not None and cols is not None:
            self.rows = rows
            self.cols = cols
            self.horizontals = [[False for _ in range(cols)] for _ in range(rows+1)]
            self.verticals = [[False for _ in range(cols+1)] for _ in range(rows)]
            self.boxes = [[None for _ in range(cols)] for _ in range(rows)]

        else:
            raise ValueError("Cannot initialize state, parameters missing")

        super().__init__(current_player, other_player)

    def get_moves(self) -> Iterable[DotsAndBoxesMove]:
        return [
            DotsAndBoxesMove("h", loc)
            for loc in self._get_free_lines(self.horizontals)
        ] + [
            DotsAndBoxesMove("v", loc)
            for loc in self._get_free_lines(self.verticals)
        ]

    def make_move(self, move: DotsAndBoxesMove) -> 'DotsAndBoxesState':
        collection = self.horizontals if move.connection == "h" else self.verticals
        if collection[move.loc[0]][move.loc[1]]:
            raise ValueError("Invalid move")

        horizontals = self._set(self.horizontals, move.loc) if move.connection == "h" else self.horizontals
        verticals = self._set(self.verticals, move.loc) if move.connection == "v" else self.verticals
        boxes, changed = self._check_boxes_after_move(horizontals, verticals, move)

        if changed:
            next_player = self._current_player
            other_player = self._other_player
        else:
            next_player = self._other_player
            other_player = self._current_player

        return DotsAndBoxesState( next_player, other_player, self.ai_depth, horizontals=horizontals, verticals=verticals, boxes=boxes)

    def is_finished(self) -> bool:
        return all(all(box_row) for box_row in self.boxes)

    def get_winner(self) -> Optional[Player]:
        if not self.is_finished():
            return None
        scores = self.get_scores()
        if scores[self._current_player] > scores[self._other_player]:
            return self._current_player
        elif scores[self._current_player] < scores[self._other_player]:
            return self._other_player
        else:
            return None

    def __str__(self) -> str:
        text = []

        for row in range(len(self.boxes)):
            text.append(self._lines_row_to_str(row))
            text.append(self._row_to_str(row))

        text.append(self._lines_row_to_str(len(self.boxes)))

        return f'Current player: {self._current_player.char}, AI: {self._current_player.ai}\n' + '\n'.join(text)

    def get_scores(self) -> Dict[Player, int]:
        scores = {
            self._current_player: 0,
            self._other_player: 0
        }

        for row in self.boxes:
            for box in row:
                if box:
                    scores[box] += 1

        return scores

    def ai_choose_move(self, moves: Iterable[DotsAndBoxesMove]) -> DotsAndBoxesMove:
        maximizing = self.get_current_player().ai

        best_score = float('-inf') if maximizing else float('inf')
        best_moves = []

        for i, move in enumerate(moves):
            new_state = self.make_move(move)
            score = self._minimax(new_state, self.ai_depth, self.get_current_player())

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

            print(i, ": ", best_moves)

        return random.choice(best_moves)

    def _minimax(self, state, depth: int, player: Player):
        if depth == 0 or state.is_finished():
            return self._evaluate()

        moves = state.get_moves()

        if player.char == '1':
            value = float('-inf')
            for move in moves:
                new_state = state.make_move(move)
                value = max(value, self._minimax(new_state, depth - 1, player))
            return value
        else:
            value = float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value = min(value, self._minimax(new_state, depth - 1, player))
            return value

    # heuristic value of node
    def _evaluate(self):
        # podstawowy wynik
        # scores = self.get_scores()
        # my_score = scores[self._current_player]
        # opp_score = scores[self._other_player]
        # zaczynamy od różnicy punktów
        # value = (my_score - opp_score) * 100

        # heurystyka pól
        three_sided_penalty = 0
        four_sided_reward = 0
        long_chain_penalty = 0

        visited = set()

        # pomocnicza funkcja do liczenia długich ścieżek
        def dfs_chain(r, c):
            stack = [(r, c)]
            length = 0
            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                # pole musi mieć mniej niż 3 boki aby tworzyć chain
                if self._count_box_sides(x, y) < 3:
                    length += 1
                    # sprawdzamy sąsiadów (góra, dół, lewo, prawo)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(self.boxes) and 0 <= ny < len(self.boxes[0]):
                            if (nx, ny) not in visited:
                                stack.append((nx, ny))
            return length

        # przechodzimy przez pola
        for r, row in enumerate(self.boxes):
            for c, owner in enumerate(row):
                sides = self._count_box_sides(r, c)

                if sides == 3:
                    # kara za 3 ściany (dawanie punktu przeciwnikowi)
                    three_sided_penalty += 20

                elif sides == 4 and owner == self._current_player:
                    # duża nagroda za zamknięcie pudełka
                    four_sided_reward += 30

                # wykrywanie długiej ścieżki
                if sides < 3 and (r, c) not in visited:
                    chain_len = dfs_chain(r, c)
                    if chain_len >= 3:
                        # kara za stworzenie długiego łańcucha
                        long_chain_penalty += chain_len * 5

        value = four_sided_reward - three_sided_penalty - long_chain_penalty
        return value

    def _get_free_lines(self, collection: List[List[bool]]) -> List[Tuple[int, int]]:
        return [
            (loc1, loc2)
            for loc1, subcol in enumerate(collection)
            for loc2, line in enumerate(subcol)
            if not line
        ]

    def _set(self, collection: List[List[bool]], loc: Tuple[int, int]) -> List[List[bool]]:
        return [
            [
                True if loc2 == loc[1] else line
                for loc2, line in enumerate(subcol)
            ] if loc1 == loc[0] else subcol
            for loc1, subcol in enumerate(collection)
        ]

    def _check_box(self, horizontals, verticals, col, row) -> Player | None:
        if self.boxes[row][col]:
            return self.boxes[row][col]

        if horizontals[col][row] and horizontals[col][row + 1] and verticals[row][col] and verticals[row][col + 1]:
            return self._current_player

        return None

    def _check_boxes_after_move( self, horizontals: List[List[bool]], verticals: List[List[bool]], move: DotsAndBoxesMove) -> Tuple[List[List[Player]], bool]:
        box1_col = None
        box1_row = None        

        box2_col = None
        box2_row = None

        if move.connection == "h":
            col, row = move.loc
            if row > 0:
                box1_col = col
                box1_row = row - 1
            if row < len(self.boxes):
                box2_col = col
                box2_row = row
        else:
            row, col = move.loc
            if col > 0:
                box1_col = col - 1
                box1_row = row
            if col < len(self.boxes[0]):
                box2_col = col
                box2_row = row

        new_boxes = [
            [
                self._check_box(horizontals, verticals, col, row)
                if col in (box1_col, box2_col) else box
                for col, box in enumerate(row_boxes)
            ] if row in (box1_row, box2_row) else row_boxes
            for row, row_boxes in enumerate(self.boxes)
        ]

        changed = False
        if box1_col is not None:
            if self.boxes[box1_row][box1_col] != new_boxes[box1_row][box1_col]:
                changed = True

        if box2_col is not None:
            if self.boxes[box2_row][box2_col] != new_boxes[box2_row][box2_col]:
                changed = True

        return new_boxes, changed

    def _count_box_sides(self, row: int, col: int) -> int:
        """Return number of drawn sides around box at (row, col)."""
        count = 0

        # Top horizontal: horizontals[col][row]
        if self.horizontals[col][row]:
            count += 1

        # Bottom horizontal: horizontals[col][row + 1]
        if self.horizontals[col][row + 1]:
            count += 1

        # Left vertical: verticals[row][col]
        if self.verticals[row][col]:
            count += 1

        # Right vertical: verticals[row][col + 1]
        if self.verticals[row][col + 1]:
            count += 1

        return count

    def _lines_row_to_str(self, row):
        line_row = self.horizontals[row]
        line_chars = ('-' if line else ' ' for line in line_row)
        return 'o' + 'o'.join(line_chars) + 'o'

    def _row_to_str(self, row):
        chars = []
        for line, box in zip(self.verticals[row], self.boxes[row]):
            chars.append('|' if line else ' ')
            chars.append(box.char if box else ' ')

        chars.append('|' if self.verticals[row][-1] else ' ')

        return ''.join(chars)
