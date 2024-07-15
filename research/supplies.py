import time

import numpy as np

from research import result


def transform_for_model(puzzle):
    return np.array(puzzle).reshape(-1, 9, 9)


def transform_from_model(result):
    data_array = np.array(result)
    data_array = np.squeeze(data_array, axis=0)
    data_int = data_array.astype(int)

    return data_int.tolist()


def count_accuracy(solution, prediction):
    counter = 0
    for row1, row2 in zip(solution, prediction):
        for elem1, elem2 in zip(row1, row2):
            if elem1 != elem2:
                counter += 1

    return (81-counter)/81.00 * 100


def get_result_from_model(model, sudoku, model_name):
    start = time.time()
    prediction = model.predict(transform_for_model(sudoku.puzzle))
    end = time.time()
    return result.ModelResult(sudoku.sudoku_id, model_name, end - start, sudoku.difficulty, count_accuracy(sudoku.solution, transform_from_model(prediction)))

