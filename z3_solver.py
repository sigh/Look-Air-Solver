"""
Solver for Look-Air puzzles.

Usage: python3 z3_solver.py < puzzle.txt

Requires z3 to be install: pip install z3-solver
"""

import copy
import fileinput
from z3 import Solver, Int, Sum, Or, If, And, sat


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

    def all_coords(self) -> list[tuple[int, int]]:
        return [(row, col) for row in range(self.height) for col in range(self.width)]

    def coord_is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def _render_value(self, value: int, clue: str) -> str:
        if value == Grid.EMPTY_VALUE:
            return clue

        return f"\x1b[1;37;40m{clue}\x1b[0m"

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


def make_solver(puzzle: LookAirPuzzle) -> None:
    s = Solver()

    grid = Grid(puzzle.grid_size)
    square_size = {}

    coords = grid.all_coords()
    # Create padded coordinates to simplify edge cases.
    padded_coords = [
        (row, col)
        for row in range(-1, grid.height + 1)
        for col in range(-1, grid.width + 1)
    ]

    # Every cell is empty or part of a shaded square.
    max_square_size = min(grid.height, grid.width)
    for coord in padded_coords:
        symbol = Int(str(coord))
        square_size[coord] = symbol
        if grid.coord_is_valid(*coord):
            s.add(0 <= symbol, symbol <= max_square_size)
        else:
            # padding is always 0
            s.add(symbol == 0)

    # The count of shaded cells matches the clues.
    for (row, col), clue in puzzle.clues.items():
        adj_coords = [
            (row + dr, col + dc)
            for dr, dc in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
            if grid.coord_is_valid(row + dr, col + dc)
        ]
        s.add(Sum([(square_size[c] > 0) for c in adj_coords]) == clue)

    # Create offsets for tracking the position within each square.
    offsets_col = {}
    offsets_row = {}
    for coord in padded_coords:
        offsets_col[coord] = Int(f"offset_col_{coord}")
        offsets_row[coord] = Int(f"offset_row_{coord}")
        # Padding is always 0
        if not grid.coord_is_valid(*coord):
            s.add(offsets_col[coord] == 0)
            s.add(offsets_row[coord] == 0)

    # Offsets are within the square size (or 0 outside squares)
    for coord in coords:
        row, col = coord
        s.add(
            If(
                square_size[coord] > 0,
                And(0 < offsets_col[coord], offsets_col[coord] <= square_size[coord]),
                offsets_col[coord] == 0,
            )
        )
        s.add(
            If(
                square_size[coord] > 0,
                And(0 < offsets_row[coord], offsets_row[coord] <= square_size[coord]),
                offsets_row[coord] == 0,
            )
        )

    # Offsets increase in value, except at the far edge of a square.
    # Square size is constant within a square.
    for coord in padded_coords:
        row, col = coord
        if row >= 0 and col >= 0:
            prev = (row - 1, col)
            s.add(
                If(
                    offsets_col[coord] > 0,
                    offsets_col[prev] + 1 == offsets_col[coord],
                    offsets_col[prev] == square_size[prev],
                )
            )
            s.add(
                If(
                    offsets_col[coord] > 1,
                    square_size[prev] == square_size[coord],
                    True,
                )
            )

            prev = (row, col - 1)
            s.add(
                If(
                    offsets_row[coord] > 0,
                    offsets_row[prev] + 1 == offsets_row[coord],
                    offsets_row[prev] == square_size[prev],
                )
            )
            s.add(
                If(
                    offsets_row[coord] > 1,
                    square_size[prev] == square_size[coord],
                    True,
                )
            )

    # Squares of the same size can't see each other.
    # This means that if two cells in the same row/col have the same square size
    # there must be a shaded cell between them.
    #
    # NOTE:
    #  - There must be at least a cell between two squares so we can start
    #    from i+1
    #  - By only checking when offset_* == 1 we only constrain the boundary of
    #    the square.
    for row in range(grid.height):
        for i in range(grid.width):
            for j in range(i + 2, grid.width):
                s.add(
                    If(
                        And(
                            offsets_row[(row, j)] == 1,
                            square_size[(row, i)] == square_size[(row, j)],
                        ),
                        Or([square_size[(row, k)] > 0 for k in range(i + 1, j)]),
                        True,
                    )
                )
    for col in range(grid.width):
        for i in range(grid.height):
            for j in range(i + 2, grid.height):
                s.add(
                    If(
                        And(
                            offsets_col[(j, col)] == 1,
                            square_size[(i, col)] == square_size[(j, col)],
                        ),
                        Or([square_size[(k, col)] > 0 for k in range(i + 1, j)]),
                        True,
                    )
                )

    return s


def model_to_grid(puzzle, model, prefix=""):
    decl_lookup = dict((d.name(), model[d]) for d in model.decls())
    grid = Grid(puzzle.grid_size)

    for row, col in grid.all_coords():
        value = decl_lookup[prefix + str((row, col))]
        grid.array[row][col] = value.as_long()

    return grid


if __name__ == "__main__":
    input_str = "\n".join(fileinput.input())
    puzzle = LookAirPuzzle.from_string(input_str)

    s = make_solver(puzzle)
    if s.check() == sat:
        model_to_grid(puzzle, s.model()).print_with_clues(puzzle.clues)
    else:
        print("unsolvable")
