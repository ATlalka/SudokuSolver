def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[i + row][j + col] == num:
                return True
    return False


def check_location_is_safe(arr, row, col, num):
    return (not used_in_row(arr, row, num) and
            (not used_in_col(arr, col, num) and
             (not used_in_box(arr, row - row % 3,
                              col - col % 3, num))))


def compare(puzzle, solution):
    for i in range(len(puzzle)):
        if len(puzzle[i]) != len(solution[i]):
            return False

        for j in range(len(puzzle[i])):
            if puzzle[i][j] != solution[i][j]:
                return False

    return True


def print_grid(grid):
    output_puzzle = ""
    for row in grid:
        output_puzzle += " ".join(map(str, row)) + "\n"

    print(output_puzzle)
