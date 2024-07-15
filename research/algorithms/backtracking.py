import copy
import time

from research.algorithms.common import check_location_is_safe, compare
from research.result import Result


# Function to Find the entry in
# the Grid that is still  not used
# Searches the grid to find an
# entry that is still unassigned. If
# found, the reference parameters
# row, col will be set the location
# that is unassigned, and true is
# returned. If no unassigned entries
# remains, false is returned.
# 'l' is a list  variable that has
# been passed from the solve_sudoku function
# to keep track of incrementation
# of Rows and Columns
def find_empty_location(grid, l):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


# Takes a partially filled-in grid
# and attempts to assign values to
# all unassigned locations in such a
# way to meet the requirements
# for Sudoku solution (non-duplication
# across rows, columns, and boxes)
def solve_sudoku(grid):
    # 'l' is a list variable that keeps the
    # record of row and col in
    # find_empty_location Function
    l = [0, 0]

    # If there is no unassigned
    # location, we are done
    if not find_empty_location(grid, l):
        return True

    # Assigning list values to row and col
    # that we got from the above Function
    row = l[0]
    col = l[1]

    # consider digits 1 to 9
    for num in range(1, 10):

        # if looks promising
        if (check_location_is_safe(grid,
                                   row, col, num)):

            # make tentative assignment
            grid[row][col] = num

            # return, if success,
            # ya !
            if solve_sudoku(grid):
                return True

            # failure, unmake & try again
            grid[row][col] = 0

    # this triggers backtracking
    return False


def solve(sudoku, with_result=False):
    sudoku_copy = copy.copy(sudoku)

    start = time.time()
    solve_sudoku(sudoku.puzzle)
    end = time.time()

    if with_result:
        sudoku_copy.print_puzzle()
        print('\n')
        sudoku.print_solution()

    if compare(sudoku.puzzle, sudoku.solution):
        # print(f"ID: {sudoku.sudoku_id}, difficulty: {sudoku.difficulty}, time: {end - start}")
        return Result(sudoku.sudoku_id, 'backtracking', end - start, sudoku.difficulty)

    else:
        # print(f"For ID {sudoku.sudoku_id} given solution doesn't match with original one. Time: {end - start}")
        return Result(sudoku.sudoku_id, 'backtracking', -1, sudoku.difficulty)
