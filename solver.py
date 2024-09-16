from typing import Dict, Tuple, List
from collections import Counter
import copy

class LookAirPuzzle:
    def __init__(self, grid_size: Tuple[int, int], clues: Dict[Tuple[int, int], int]):
        self.grid_size = grid_size
        self.clues = clues

    def print(self) -> None:
        Grid(self.grid_size).print_with_clues(self.clues)

    def from_string(s: str) -> 'LookAirPuzzle':
      lines = [l.strip() for l in s.strip().split('\n')]
      grid_size = (len(lines), len(lines[0]))
      clues = {}
      for (row, line) in enumerate(lines):
        for (col, char) in enumerate(line):
          if char.isdigit():
            clues[(row, col)] = int(char)
      return LookAirPuzzle(grid_size=grid_size, clues=clues)

class Grid:
  UNSET_VALUE = -1
  EMPTY_VALUE = 0

  def __init__(self, size: Tuple[int, int]):
    self.height, self.width = size
    self.array = [ [Grid.UNSET_VALUE] * self.width for _ in range(self.height) ]
    self.num_cells = self.width * self.height

  def copy(self):
    return copy.deepcopy(self)

  def set_square(self, row: int, col: int, square_size: int, value: int) -> None:
    self.array[row][col] = value
    for i in range(row, row + square_size):
      for j in range(col, col + square_size):
        self.array[i][j] = value

  def all_coords(self) -> List[Tuple[int, int]]:
      return [(row, col) for row in range(self.height) for col in range(self.height)]

  def print_with_clues(self, clues: Dict[Tuple[int, int], int]) -> None:
      print('-' * self.width)
      for row, cells in enumerate(self.array):
          chars = [str(clues.get((row, col), ' ')) for col in range(len(cells))]
          for col, v in enumerate(cells):
            if v > Grid.EMPTY_VALUE:
              chars[col] = '\x1b[1;37;40m' + chars[col] + '\x1b[0m'
            elif v == Grid.UNSET_VALUE:
              chars[col] = '\x1b[1;37;44m' + chars[col] + '\x1b[0m'
          print(''.join(chars))
      print('-' * self.width)

  def print_debug(self) -> None:
      print('-' * self.width)
      for row in self.array:
          print(''.join(str(v) if v != Grid.UNSET_VALUE else ' ' for v in row))
      print('-' * self.width)

class LookAirSolver:

  def __init__(self, puzzle: LookAirPuzzle):
    self.puzzle = puzzle

  def _is_possible_square(self, grid: Grid, row: int, col: int, square_size: int) -> bool:
    if square_size == 0:
      return True

    array = grid.array

    # Every cell must be unset
    for i in range(row, row + square_size):
      for j in range(col, col + square_size):
        if array[i][j] != Grid.UNSET_VALUE:
          return False

    # There must be no adjacent squares
    for i in range(row, row + square_size):
      if col > 0 and array[i][col - 1] > Grid.EMPTY_VALUE:
        return False
      if col + square_size < grid.width and array[i][col + square_size] > Grid.EMPTY_VALUE:
        return False
    for j in range(col, col + square_size):
      if row > 0 and array[row - 1][j] > Grid.EMPTY_VALUE:
        return False
      if row + square_size < grid.height and array[row + square_size][j] > Grid.EMPTY_VALUE:
        return False

    # The squares can't see a square of the same size
    for i in range(row, row + square_size):
      if self._square_in_direction(grid, i, col-1, 0, -1) == square_size:
        return False
    for j in range(col, col + square_size):
      if self._square_in_direction(grid, row-1, j, -1, 0) == square_size:
        return False

    return True

  def _validate_affected_cells(self, grid: Grid, row: int, col: int, square_size: int) -> bool:
    # Check the clues in and around the current square
    for i in range(row - 1, row + square_size + 1):
      for j in range(col - 1, col + square_size + 1):
        if not self._is_valid_clue(grid, i, j):
          return False

    # Check that the visibility rules are still respected.
    #   This is required because new squares may be added after a possible
    #   square is placed.
    # We only need to do this is one direction, but try all so that we aren't
    # dependant on the order of the solver.
    if square_size == 0:
      row_target = self._square_in_direction(grid, row, col, 1, 0)
      if row_target is not None and row_target == self._square_in_direction(grid, row, col, -1, 0):
        return False
      col_target = self._square_in_direction(grid, row, col, 0, 1)
      if col_target is not None and col_target == self._square_in_direction(grid, row, col, 0, -1):
        return False

    return True

  def _square_in_direction(self, grid: Grid, row: int, col: int, dx: int, dy: int) -> int:
    while 0 <= row < grid.height and 0 <= col < grid.width:
      v = grid.array[row][col]
      if v > Grid.EMPTY_VALUE:
        return v
      if v == Grid.UNSET_VALUE:
        return None
      row += dx
      col += dy
    return None

  def _is_valid_clue(self, grid: Grid, row: int, col: int) -> bool:
    if (row, col) not in self.puzzle.clues:
      return True

    array = grid.array

    counter = Counter()
    counter[array[row][col]] += 1
    if row > 0:
      counter[array[row - 1][col]] += 1
    if col > 0:
      counter[array[row][col - 1]] += 1
    if row < grid.height - 1:
      counter[array[row + 1][col]] += 1
    if col < grid.width - 1:
      counter[array[row][col + 1]] += 1

    target = self.puzzle.clues[(row, col)]
    count_unset = counter[Grid.UNSET_VALUE]
    count_empty = counter[Grid.EMPTY_VALUE]
    count_shaded = counter.total() - count_empty - count_unset
    if count_shaded > target:
      return False
    if count_shaded + count_unset < target:
      return False
    return True

  def _index_to_coords(self, index: int, grid) -> Tuple[int, int]:
    return index // grid.width, index % grid.width

  def solve(self, current_index: int = 0, grid: Grid = None):
    if grid is None:
      grid = Grid(self.puzzle.grid_size)

    # Find next unset index
    while current_index < grid.num_cells:
      row, col = self._index_to_coords(current_index, grid)
      if grid.array[row][col] == Grid.UNSET_VALUE:
         break
      current_index += 1

    # Check if we found a solution
    if current_index == grid.num_cells:
      yield grid.copy()
      return

    row, col = self._index_to_coords(current_index, grid)

    # Try all possible square sizes
    for square_size in range(min(grid.height - row, grid.width - col)+1):
      if not self._is_possible_square(grid, row, col, square_size):
        continue

      grid.set_square(row, col, square_size, square_size)

      if self._validate_affected_cells(grid, row, col, square_size):
        for solution in self.solve(current_index + 1, grid):
          yield solution

      grid.set_square(row, col, square_size, Grid.UNSET_VALUE)

if __name__ == '__main__':
  # https://www.cross-plus-a.com/html/cros7rukk.htm
  puzzle1 = LookAirPuzzle.from_string(
    """
      1........1
      0.2......2
      ..........
      .1..1..3..
      ..........
      ..........
      ..2..3..3.
      ..........
      1......2.1
      2........1
    """)

  # https://www.youtube.com/watch?v=wIvjsPrWUSA
  puzzle2 = LookAirPuzzle.from_string(
    """
      ................
      .5..3..21...12..
      .4..1.3..3.3..2.
      .1..1.3..2.3..3.
      ..121.1..3.1..3.
      ....2.1..3.1..1.
      ....2.4..1.1..2.
      ....2..41...11..
      ................
    """)

  puzzle = puzzle1

  puzzle.print()
  solver = LookAirSolver(puzzle)
  for solution in solver.solve():
      solution.print_with_clues(puzzle.clues)