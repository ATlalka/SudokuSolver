class Result:
    def __init__(self, sudoku_id, algorithm, time, difficulty):
        self.sudoku_id = sudoku_id
        self.algorithm = algorithm
        self.time = time
        self.difficulty = difficulty

    def to_csv_row(self):
        return [self.algorithm, self.sudoku_id, self.difficulty, self.time]


class ModelResult:
    def __init__(self, sudoku_id, algorithm, time, difficulty, accuracy):
        self.sudoku_id = sudoku_id
        self.algorithm = algorithm
        self.time = time
        self.difficulty = difficulty
        self.accuracy = accuracy

    def to_csv_row(self):
        return [self.algorithm, self.sudoku_id, self.difficulty, self.time, self.accuracy]