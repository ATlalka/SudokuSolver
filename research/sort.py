import csv

from research import reader


def sort():
    sudokus = reader.read_all()
    sudokus.sort(key=lambda x: x.difficulty)
    difficulties = []
    sorted = []

    for i in range(0, 86):
        difficulties.append(i / 10)

    while len(sorted) < 1001:
        for d in difficulties:
            for s in sudokus:
                if s.difficulty == d:
                    sorted.append(s)
                    sudokus.remove(s)
                    with open('sudoku-3m-sorted.csv.', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(s.to_csv_row())
                    break
