from research.sudoku import Sudoku


def read_by_difficulty(min_range=0.0, max_range=8.5):
    result = []
    f = open("../dataset/sudoku-3m.csv", "r")
    f.readline()  # skip first line with headers
    for line in f:
        current = convert_to_object(line)
        if max_range >= current.difficulty >= min_range:
            result.append(current)

    return result


def read_by_id(sudoku_id):
    f = open("../dataset/sudoku-3m.csv", "r")
    for _ in range(sudoku_id):
        f.readline()

    return convert_to_object(f.readline())


def read_all():
    result = []
    f = open("././dataset/sudoku-3m-sorted.csv", "r")
    f.readline()  # skip first line with headers
    for line in f:
        current = convert_to_object(line)
        result.append(current)

    return result


def convert_to_object(line):
    data = line.split(",")
    puzzle = []
    solution = []

    for i in range(0, len(data[1]), 9):
        row = data[1][i:i + 9]
        row = row.replace('.', '0')
        puzzle.append([int(number) for number in row])

    for i in range(0, len(data[2]), 9):
        row = data[2][i:i + 9]
        row = row.replace('.', '0')
        solution.append([int(number) for number in row])

    sudoku = Sudoku(int(data[0]), puzzle, solution, int(data[3]), float(data[4]))

    return sudoku
