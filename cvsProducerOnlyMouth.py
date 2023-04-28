# -----------------------------------------------------------------------------
# Script che si occupa di prendere video da 15 secondi con un viso presente
# e per ogni fotogramma del video, isola le labbra rendendole della stessa
# dimensione e stampa le distanze euclidee.
# Inoltre stampa una copia del video in cui è visibile solo il riquadro delle
# labbra, ed una copia in cui stampa a video anche i landmark
# Nb. La stampa del video senza landmark è attualmente commentata, alla riga 166
#
# Mario Paone, Emanuele Mezzi
# ------------------------------------------------------------------------------

import cv2
import os
import numpy as np
import dlib
import csv
from collections import OrderedDict

# Inizializza il face detector(HOG-based) della libreria dlib.
detector = dlib.get_frontal_face_detector()

# Crea il predictor per i landmark del viso.
predictor = dlib.shape_predictor("/Users/memex_99/desktop/Tesi/shape_predictor_68_face_landmarks.dat")

# Path della cartella contenente i video da 15 secondi da computare. 
path = "/Users/memex_99/desktop/15_sec_giapp"

# Path della cartella di destinazione per i video csv. 
destinationPath = "/Users/memex_99/desktop/nuovi_csv"

# path della cartella di destinazione per i video con presenti solo le labbra. 
video_destination_path = "/Users/memex_99/desktop/nuovi_solo"

# path della cartella di destinazione per i video con presenti i landmark sulle labbra. 
video_destination_path_land = "/Users/memex_99/desktop/nuovi_land"

# Cambia la cartella di destinazione in quella di destinationPath. 
os.chdir(destinationPath)

# Dimensione a cui verranno stampati i video della labbra.
SIZE = (300, 200)

# Definisce un dizionario che mappa gli indici per i landmark facciali per ogni regione del viso.
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("mouth_intern", (60, 68)),
    ("mouth_extern", (48, 60)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

"""
Questa funzione prende un array di fotogrammi contigui, il nome del video 
e il path di destinazione e stampa il rispettivo video. 
"""


def save_video(array_video, name, where_to_save):
    # Cambia il path della destinazione.
    os.chdir(where_to_save)
    print("Produco video " + name)

    # Salva il framerate del video.
    framerate = name.split('_')[4]
    print("Il framerate è: " + str(framerate))

    # Se l'array non è vuoto. 
    if len(array_video) > 0:

        # Apre il video da salvare.
        out = cv2.VideoWriter(str(name) + "_m" + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), int(framerate), SIZE)
        for frame in array_video:
            # Aggiunge il frame al video
            out.write(frame)

        # Rilascia il video e lo salva definitivamente. 
        out.release()
        print("Completata computazione video " + str(name))


"""
Questa funzione prende una stringa nome ed una matrice
di distanze euclidee e stampa su file .csv la matrice. 
"""


def print_csv_file(filename, matrix):
    os.chdir(destinationPath)

    # Apre il file csv.
    with open(str(filename) + ".csv", mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Per ogni frame del video scrive le distanze euclidee nel file.
        for row in matrix:
            writer.writerow(row)


"""
Restituisce le una lista contenente le coppie
di coordinate che rappresentano i landmark. 
"""


def shape_to_np(shape_par, dtype="int"):
    # Inizializza la lista.
    coords = np.zeros((68, 2), dtype=dtype)

    for j in range(0, 68):
        coords[j] = (shape_par.part(j).x, shape_par.part(j).y)

    return coords


# Per ogni file video nella cartella. 
for videoFile in os.listdir(path):

    if videoFile != ".DS_Store":

        print("-----------Inizio computazione " + videoFile + "----------------")

        # Apre il video
        cap = cv2.VideoCapture(path + "/" + videoFile)

        # Conterrà N array ognuno contenente le distanze euclidee per ogni singolo frame. 
        distance_matrix_ext = []

        # Conterrà i singoli frame delle labbra che verranno usati per formare il video. 
        video_array = []

        # Conterrà i singoli frame delle labbra con i landmark stampati a video usati per formare il video. 
        video_array_landmark = []

        # Fin quando il video non sarà concluso.
        s = 0
        while cap.isOpened():

            print("Frame: ", s)
            s = s + 1

            # Salva ogni frame in image
            ret, image = cap.read()

            if not ret:
                break

            # image = cv2.resize(image, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)
            # Estrae i rettangoli contenenti visi. 
            rects = detector(image, 1)

            # Per ogni rettangolo contenente un viso.
            for rect in rects:

                # Determina i landmark del viso. 
                shape = predictor(image, rect)

                # Converte i landmark in coordinate (x, y) in un array NumPy. 
                shape = shape_to_np(shape)

                # Array delle distanze euclidee per i singoli frame. 
                distance_list = []

                # Estrae i punti per il rettangolo contenente le labbra
                # x e y sono le coordinate del vertice in alto a sinistra, 
                # mentre w e h sono la larghezza e l'altezza. 
                (x, y, w, h) = cv2.boundingRect(np.array([shape[FACIAL_LANDMARKS_IDXS["mouth"][0]:
                                                                FACIAL_LANDMARKS_IDXS["mouth"][1]]]))

                # Estrae il rettangolo contenente le labbra(Con dimensione 10 in più da ogni lato). 
                roi = image[y - 10:y + h + 10, x - 10:x + w + 10]

                # Da' al frame dimensione SIZE.
                new = cv2.resize(roi, dsize=SIZE, interpolation=cv2.INTER_CUBIC)

                # Aggiunge il frame all'array dei frame del video. 
                video_array.append(new)

                # Prende solo i landmark per le labbra. 
                xm, ym = FACIAL_LANDMARKS_IDXS["mouth_intern"]
                new_shape = []

                # Copia l'immagine. 
                new_copy = np.copy(new)

                # Per ogni coppia di coordinate scelta. 
                for (xa, ya) in shape[xm:ym]:
                    # In queste 4 righe di codice prende i landmark dell'immagine originale e li
                    # scala in modo da adattarli alle nuove dimensioni scandite dalla
                    # costante SIZE. 
                    xi = xa - (x - 10)
                    yi = ya - (y - 10)
                    new_x = int((xi * SIZE[0]) / (w + 20))
                    new_y = int((yi * SIZE[1]) / (h + 20))

                    # Stampa i landmark sull'immagine. 
                    cv2.circle(new_copy, (new_x, new_y), 1, (0, 0, 255), -1)

                    # Aggiunte le coordiante dei frame all'array. 
                    new_shape.append((new_x, new_y))

                # Aggiunge le immagini all'array per stampare il video. 
                video_array_landmark.append(new_copy)

                distance_list = [int(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])))
                                 for i, (x1, y1) in enumerate(new_shape)
                                 for (x2, y2) in new_shape[i + 1:]]

                distance_matrix_ext.append(distance_list)

        # Chiama la funzione per stampare il video della labbra senza landmark.
        # save_video(video_array, videoFile.split('.')[0], video_destination_path)

        # Chiama la funzione per stampare il video della labbra con i landmark. 
        save_video(video_array_landmark, videoFile.split('.')[0], video_destination_path_land)

        # Chiama la funzione per stampare la matrice di distanze nell'omonimo file csv. 
        print_csv_file(videoFile.split(".")[0] + "_m", distance_matrix_ext)

        print("-----------Conclusa computazione " + videoFile + "----------------")

cap.release()
cv2.destroyAllWindows()
