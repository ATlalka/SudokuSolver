import copy
import time

from research.algorithms import backtracking
from research.algorithms.common import compare
from research.result import Result


def solve_sudoku(grid):
    was_found = True
    while was_found:
        was_found = False
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    candidates = set(range(1, 10))
                    for x in range(9):
                        candidates.discard(grid[i][x])
                        candidates.discard(grid[x][j])
                    start_row, start_col = 3 * (i // 3), 3 * (j // 3)
                    for x in range(start_row, start_row + 3):
                        for y in range(start_col, start_col + 3):
                            candidates.discard(grid[x][y])
                    if len(candidates) == 1:
                        grid[i][j] = candidates.pop()
                        was_found = True


def solve(sudoku, with_result=False):
    sudoku_copy = copy.copy(sudoku)

    start = time.time()
    solve_sudoku(sudoku.puzzle)
    backtracking.solve_sudoku(sudoku.puzzle)
    end = time.time()

    if with_result:
        sudoku_copy.print_puzzle()
        print('\n')
        sudoku.print_solution()

    if compare(sudoku.puzzle, sudoku.solution):
        # print(f"ID: {sudoku.sudoku_id}, difficulty: {sudoku.difficulty}, time: {end - start}")
        return Result(sudoku.sudoku_id, 'single candidate + backtracking', end - start, sudoku.difficulty)

    else:
        # print(f"For ID {sudoku.sudoku_id} given solution doesn't match with original one. Time: {end - start}")
        return Result(sudoku.sudoku_id, 'single candidate + backtracking', -1, sudoku.difficulty)
