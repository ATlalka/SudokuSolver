import copy
import time

from research.algorithms import backtracking
from research.algorithms.common import check_location_is_safe, compare
from research.result import Result


def check_row_positions(num, grid, row):
    position_candidates = []
    for i in range(9):
        if grid[row][i] == 0 and check_location_is_safe(grid, row, i, num):
            position_candidates.append(i)

    if len(position_candidates) == 1:
        location = position_candidates.pop()
        grid[row][location] = num
        return True

    return False


def check_col_positions(num, grid, col):
    position_candidates = []
    for i in range(9):
        if grid[i][col] == 0 and check_location_is_safe(grid, i, col, num):
            position_candidates.append(i)

    if len(position_candidates) == 1:
        location = position_candidates.pop()
        grid[location][col] = num
        return True

    return False


def check_box_positions(num, grid, row, col):
    position_candidates = []
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for x in range(start_row, start_row + 3):
        for y in range(start_col, start_col + 3):
            if grid[x][y] == 0 and check_location_is_safe(grid, x, y, num):
                position_candidates.append(x)
                position_candidates.append(y)

    if len(position_candidates) == 2:
        location_y = position_candidates.pop()
        location_x = position_candidates.pop()
        grid[location_x][location_y] = num
        return True

    return False


def solve_sudoku(grid):
    was_found = True
    while was_found:
        was_found = False
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for x in range(1, 10):
                        if check_row_positions(x, grid, i) or check_col_positions(x, grid, j) or check_box_positions(x,
                                                                                                                     grid,
                                                                                                                     i,
                                                                                                                     j):
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
        return Result(sudoku.sudoku_id, 'single position + backtracking', end - start, sudoku.difficulty)

    else:
        # print(f"For ID {sudoku.sudoku_id} given solution doesn't match with original one. Time: {end - start}")
        return Result(sudoku.sudoku_id, 'single position + backtracking', -1, sudoku.difficulty)
