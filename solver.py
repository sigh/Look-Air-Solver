"""
Solver for Look-Air puzzles.

Usage: python3 solver.py < puzzle.txt

The puzzle consists of a grid of cells, some of which contain numbers. The
goal is to blacken the cells such that:

- Black cells form an area of square shape.
- No black areas touch each other horizontally or vertically (only diagonally).
- Black areas of the same size must not "see" each other: horizontal or
  vertical lines between two black areas must contain at least one black area
  of other size.
- Each number shows how many of the five cells (the one with the number plus
  the four orthogonally neighboring cells) should be blacken.
"""

import copy
import fileinput
from typing import Optional


class LookAirPuzzle:
    def __init__(self, grid_size: tuple[int, int], clues: dict[tuple[int, int], int]):
        self.grid_size = grid_size
        self.clues = clues

    def print(self) -> None:
        Grid(self.grid_size).print_with_clues(self.clues)

    def from_string(s: str) -> "LookAirPuzzle":
        lines = [l.strip() for l in s.strip().split("\n")]
        lines = [l for l in lines if l and not l.startswith("#")]
        grid_size = (len(lines), len(lines[0]))
        clues = {}
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if char.isdigit():
                    clues[(row, col)] = int(char)
        return LookAirPuzzle(grid_size=grid_size, clues=clues)


class Grid:
    """Grid represents the Look-Air puzzle solution or partial solution

    It contains a height*width 2D-array. The values are:
     - UNSET_VALUE (-1) if the cell is not yet set.
     - EMPTY_VALUE (0) if the cell is set to empty.
     - Any number (1..) if the cell is part of a shaded square. The value gives
       the size of the square to make checking constraints easier.
    """

    UNSET_VALUE = -1
    EMPTY_VALUE = 0

    def __init__(self, size: tuple[int, int]):
        self.height, self.width = size
        self.array = [[Grid.UNSET_VALUE] * self.width for _ in range(self.height)]
        self.num_cells = self.width * self.height

    def copy(self) -> "Grid":
        return copy.deepcopy(self)

    def set_square(self, row: int, col: int, square_size: int, value: int) -> None:
        self.array[row][col] = value
        for i in range(row, row + square_size):
            for j in range(col, col + square_size):
                self.array[i][j] = value

    def all_coords(self) -> list[tuple[int, int]]:
        return [(row, col) for row in range(self.height) for col in range(self.height)]

    def index_to_coords(self, index) -> tuple[int, int]:
        """Convert a cell index to grid coordinates"""
        return index // self.width, index % self.width

    def coord_is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def _render_value(self, value: int, clue: str) -> str:
        if value == Grid.EMPTY_VALUE:
            return clue

        bg_color = 44 if value == Grid.UNSET_VALUE else 40
        return f"\x1b[1;37;{bg_color}m{clue}\x1b[0m"

    def print_with_clues(self, clues: dict[tuple[int, int], int]) -> None:
        print("-" * self.width)
        for row, cells in enumerate(self.array):
            chars = [str(clues.get((row, col), " ")) for col in range(len(cells))]
            chars = [self._render_value(v, c) for (v, c) in zip(cells, chars)]
            print("".join(chars))
        print("-" * self.width)

    def print_debug(self) -> None:
        print("-" * self.width)
        for row in self.array:
            print("".join(str(v) if v != Grid.UNSET_VALUE else " " for v in row))
        print("-" * self.width)


class LookAirSolver:
    def __init__(self, puzzle: LookAirPuzzle):
        self.puzzle = puzzle
        self.num_guesses = 0

    def _is_possible_square(
        self, grid: Grid, row: int, col: int, square_size: int
    ) -> bool:
        """Check if it is possible to place a square, ignoring puzzle clues"""

        # An empty cell is always possible.
        if square_size == 0:
            return True

        array = grid.array

        # Every cell must be unset.
        for i in range(row, row + square_size):
            for j in range(col, col + square_size):
                if array[i][j] != Grid.UNSET_VALUE:
                    return False

        # There must be no adjacent squares.
        for i in range(row, row + square_size):
            for j in [col - 1, col + square_size]:
                if grid.coord_is_valid(i, j) and array[i][j] > Grid.EMPTY_VALUE:
                    return False
        for j in range(col, col + square_size):
            for i in [row - 1, row + square_size]:
                if grid.coord_is_valid(i, j) and array[i][j] > Grid.EMPTY_VALUE:
                    return False

        # The squares can't see a square of the same size.
        # We only need to check to the left and above (i.e. already placed cells).
        for i in range(row, row + square_size):
            if self._square_in_direction(grid, i, col - 1, 0, -1) == square_size:
                return False
        for j in range(col, col + square_size):
            if self._square_in_direction(grid, row - 1, j, -1, 0) == square_size:
                return False

        return True

    def _validate_affected_cells(
        self, grid: Grid, row: int, col: int, square_size: int
    ) -> bool:
        """Validate the puzzle for all cells affected by an added square

        - Check all clues in and around the square.
        - Check visibility rules for squares when a new empty cell is placed.
        """

        # Check the clues in and around the current square
        for i in range(row - 1, row + square_size + 1):
            for j in range(col - 1, col + square_size + 1):
                if not self._is_valid_clue(grid, i, j):
                    return False

        # Check that the visibility rules are still respected.
        #   This is required because new squares may be added after a possible
        #   square is placed.
        # We only need to do this is one direction, but try all so that we
        # aren't dependant on the order of the solver.
        if square_size == 0:
            row_target = self._square_in_direction(grid, row, col, 1, 0)
            if row_target is not None and row_target == self._square_in_direction(
                grid, row, col, -1, 0
            ):
                return False
            col_target = self._square_in_direction(grid, row, col, 0, 1)
            if col_target is not None and col_target == self._square_in_direction(
                grid, row, col, 0, -1
            ):
                return False

        return True

    def _square_in_direction(
        self, grid: Grid, row: int, col: int, dx: int, dy: int
    ) -> Optional[int]:
        """Find the size of the first square in direction (dx, dy)

        Return the square size if it is found, otherwise None. Finding an
        UNSET_VALUE results in None as it may later be filled with a square.
        """
        while grid.coord_is_valid(row, col):
            v = grid.array[row][col]
            if v > Grid.EMPTY_VALUE:
                return v
            if v == Grid.UNSET_VALUE:
                return None
            row += dx
            col += dy
        return None

    def _is_valid_clue(self, grid: Grid, row: int, col: int) -> bool:
        """Check if the clue for the current cell is valid.

        Returns False if the clue results in a conflict, True otherwise.
        If we are unsure, then True is returned.
        """
        if (row, col) not in self.puzzle.clues:
            return True

        # Count the occurrence of each type of value.
        values = [
            grid.array[row + dr][col + dc]
            for dr, dc in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
            if grid.coord_is_valid(row + dr, col + dc)
        ]
        count_unset = sum(1 for v in values if v == Grid.UNSET_VALUE)
        count_shaded = sum(1 for v in values if v > Grid.EMPTY_VALUE)

        # Check if we don't have too many shaded cells, or not enough even if
        # using all the unset cells.
        # Note: When all cells are set, this will only be true when
        #       count_shaded == clue
        clue = self.puzzle.clues[(row, col)]
        return clue - count_unset <= count_shaded <= clue

    def solve(self, current_index: int = 0, grid: Optional[Grid] = None):
        """Recursively solve the Look-Air puzzle.

        The solver progresses across each row in turn attempting to place
        squares in all possible locations.
        At each cell we try leaving the cell empty, and then trying all squares
        with the cell as the top-left corner.
        """
        if grid is None:
            grid = Grid(self.puzzle.grid_size)

        # Find next unset index
        while current_index < grid.num_cells:
            row, col = grid.index_to_coords(current_index)
            if grid.array[row][col] == Grid.UNSET_VALUE:
                break
            current_index += 1

        # Check if we found a solution
        if current_index == grid.num_cells:
            yield grid.copy()
            return

        row, col = grid.index_to_coords(current_index)

        # Try all possible square sizes which the current cells at the top left
        # corner.
        for square_size in range(min(grid.height - row, grid.width - col) + 1):
            # Check if it is possible to place the square (without worrying
            # about the clues).
            if not self._is_possible_square(grid, row, col, square_size):
                continue

            grid.set_square(row, col, square_size, square_size)

            # Check all updated squares to ensure they haven't caused any
            # conflicts.
            if self._validate_affected_cells(grid, row, col, square_size):
                # If everything is ok, then recurse.
                self.num_guesses += 1
                for solution in self.solve(current_index + 1, grid):
                    yield solution

            grid.set_square(row, col, square_size, Grid.UNSET_VALUE)


if __name__ == "__main__":
    input_str = "\n".join(fileinput.input())
    puzzle = LookAirPuzzle.from_string(input_str)

    puzzle.print()
    solver = LookAirSolver(puzzle)
    for solution in solver.solve():
        solution.print_with_clues(puzzle.clues)
    print(f"Num guesses: {solver.num_guesses}")
