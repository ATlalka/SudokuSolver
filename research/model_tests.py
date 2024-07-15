import csv

from research import reader, supplies
import tensorflow as tf


def test_models(filename, models, models_dir):

    for m in models:
        model = tf.keras.models.load_model(models_dir + '/' + m + '.keras')
        sudokus = reader.read_all()

        for s in sudokus:
            result = supplies.get_result_from_model(model, s, m)
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result.to_csv_row())


