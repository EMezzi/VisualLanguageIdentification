import numpy as np
import csv

"""
Funzione che trasforma il file csv in una matrice e costruisce i training data
sia per il database da 8 landmark che per il database da 12 landmark
"""


def csv_function(csv_file, language, distances, landmark):
    MAX_LINES = 330
    line_count = 0
    array_csv = []

    with open(csv_file) as csv_file:  # apre il file csv

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count < MAX_LINES:  # Inserisce MAX_LINES righe nel dataset
                line_count += 1
                new_row = [float(i) for i in row]  # Converte in float i valori del file csv
                for element in new_row:
                    # Aggiunge tutti i valori allo stesso array in modo da avere nel Dataset
                    # ogni riga (NumpyArray) rappresenti un file csv
                    array_csv.append(element)

        array_csv = np.array(array_csv)
        array_csv.shape = (line_count, distances)

        landmark.append([array_csv, language])
