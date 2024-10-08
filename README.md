# Look-Air Solver

A basic implementation of a Look-Air (Rukkuea) puzzle solver.

An explanation of the puzzle can be found [here](https://www.cross-plus-a.com/html/cros7rukk.htm).

The puzzle consists of a grid of cells, some of which contain numbers. The goal
is to blacken the cells such that:

- Black cells form an area of square shape.
- No black areas touch each other horizontally or vertically (only diagonally).
- Black areas of the same size must not "see" each other: horizontal or vertical
  lines between two black areas must contain at least one black area of other
  size.
- Each number shows how many of the five cells (the one with the number plus the
  four orthogonally neighboring cells) should be blacken.

## Usage

```sh
# Recursive brute-force solver
python3 rec_solver.py < puzzles/example1.txt

# z3 SMT solver
# Requires z3: pip install z3-solver
python3 z3_solver.py < puzzles/example1.txt
```
