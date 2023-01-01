# -----------------------------------------------------------------------------
# Questo script, data la cartella contenente i file csv, estrae i dati 
# e restituisce il dataset 
#
# Mario Paone
# ------------------------------------------------------------------------------

import numpy as np
import csv
from sklearn.utils import Bunch
import os

path = "/Users/memex_99/Desktop/Tesi/8_landmark/VideoCsv/labbra_8"  # Path della cartella contenente il dataset
MAX_LINES = 350  # Numero massimo di righe del file csv che prende


def load_dataset():
    """
    Questo metodo restutuisce il dataset non normalizzato
    """

    dataset = []
    targets = []

    for file in os.listdir(path):  # per ogni file video nella cartella

        data = []
        line_count = 1

        if file != ".DS_Store":
            with open(path + "/" + file) as csv_file:  # apre il file csv

                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if line_count <= MAX_LINES:  # Inserisce MAX_LINES righe nel dataset
                        new_row = [float(i) for i in row]  # Converte in float i valori del file csv
                        for element in new_row:
                            data.append(element)  # Aggiunge tutti i valori allo stesso array in modo da avere nel Dataset
                            # ogni riga (NumpyArray) rappresenti un file csv
                        line_count += 1

                dataset.append(np.array(data))
                targets.append(int(file.split('_')[0]))  # Aggiunge il primo valore del nome come etichetta

    np_targets = np.array(targets)  # Converte in numpy array
    np_dataset = np.array(dataset, dtype=object)  # Converte in numpyarray

    return Bunch(data=np_dataset, target=np_targets)


data = load_dataset()

print(data)


