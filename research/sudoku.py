class Sudoku:
    def __init__(self, sudoku_id, puzzle, solution, clues, difficulty):
        self.sudoku_id = sudoku_id
        self.puzzle = puzzle
        self.solution = solution
        self.clues = clues
        self.difficulty = difficulty

    def __str__(self):
        output_puzzle = ""
        for row in self.puzzle:
            output_puzzle += " ".join(map(str, row)) + "\n"

        output_solution = ""
        for row in self.solution:
            output_solution += " ".join(map(str, row)) + "\n"

        return f"ID: {self.sudoku_id}\n\nPuzzle:\n{output_puzzle} \nSolution:\n{output_solution} \nDifficulty: {self.difficulty}"

    def print_puzzle(self):
        output_puzzle = ""
        for row in self.puzzle:
            output_puzzle += " ".join(map(str, row)) + "\n"

        print(output_puzzle)

    def print_solution(self):
        output_solution = ""
        for row in self.solution:
            output_solution += " ".join(map(str, row)) + "\n"

        print(output_solution)

    def to_csv_row(self):
        puzzle_formatted = ''.join([''.join([str(i) if i != 0 else '.' for i in row]) for row in self.puzzle])
        solution_formatted = ''.join([''.join([str(i) if i != 0 else '.' for i in row]) for row in self.solution])
        return [self.sudoku_id, puzzle_formatted, solution_formatted, self.clues, self.difficulty]
