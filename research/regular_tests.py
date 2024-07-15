import csv


from research import reader
from research.algorithms import single_position, single_candidate, backtracking


def test_single_position(filename):
    sudokus = reader.read_all()
    for s in sudokus:
        result = single_position.solve(s, False)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result.to_csv_row())


def test_single_candidate(filename):
    sudokus = reader.read_all()
    for s in sudokus:
        result = single_candidate.solve(s, False)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result.to_csv_row())


def test_backtracking(filename):
    sudokus = reader.read_all()
    for s in sudokus:
        result = backtracking.solve(s, False)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result.to_csv_row())
