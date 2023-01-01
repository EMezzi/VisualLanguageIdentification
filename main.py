# -----------------------------------------------------------------------------
# Questo script, dato un video in input, si occuppa di estrarre un numero
# prefissato di sottovideo di 15 secondi in cui il volto del soggetto
# sia ben visibile e in cui quest'ultimo parli
#
# Mario Paone, Emanuele Mezzi
# ------------------------------------------------------------------------------

import cv2
import os
import dlib

# Inizializza il face detector(HOG-based) della libreria dlib.
detector = dlib.get_frontal_face_detector()

# Crea il predictor per i landmark del viso.
predictor = dlib.shape_predictor("/Users/memex_99/desktop/Tesi/shape_predictor_68_face_landmarks.dat")

# Numero di secondi consecutivi necessari.
CONTIGUOUS_SECONDS = 5

# Numero di video massimo da produrre.
MAX_PER_VIDEO = 3

# Path della cartella contenente i video integrali.
path = "/Users/memex_99/desktop/daProdurre"

# Path di destinazione per i video da 15 secondi.
destinationPath = "/Users/memex_99/desktop/15_sec"

"""
Questa funzione prende una array contenente i frame contigui, una cartella di destionazione, 
il FrameRate e il nome del video e salva il numero massimo di sottovideo computabili tramite 
i frame contenuti in contigous_frame e salva i video prodotti nella cartella where_to_save
"""


def save_video(contiguous_frames, where_to_save, framerate, name):

    # Cambia la cartella di destinazione in quella di destinationPath
    os.chdir(where_to_save)

    # L'array deve contenere almeno una sequenza di 5 secondi
    if len(contiguous_frames) > 0:

        # Prende altezza e larghezza del video
        height, width = len(contiguous_frames[0][0]), len(contiguous_frames[0][0][0])

        size = (width, height)  # Setta la dimensione del video
        print("Dimensione video = " + str(size))
        print("Numero di frame al secondo = " + str(framerate))

        # Se contigous_frame contiene almeno 3 sequenze da TOT secondi ciascuna
        if len(contiguous_frames) >= 3:
            for i in range(len(contiguous_frames) // 3):

                # Crea il video
                out = cv2.VideoWriter(str(name) + "_" + str(i + 1) + ".avi",
                                      cv2.VideoWriter_fourcc(*'DIVX'), int(framerate), size)

                for j in range(i * 3, i * 3 + 3):
                    for image in contiguous_frames[j]:
                        out.write(image)

                out.release()
                print("Completata computazione video: " + str(name) + "_" + str(i + 1))


"""
Questa funzione analizza ogni frame di un video e salva in un array tutte le sequenze di frame contigue e chiama il
metodo save_video per salvare i video computabili da quelle sequenze
"""


def create_videos(path_, video, where_to_save):

    if video != ".DS_Store":

        print("Apertura file " + video)

        # Salva il frame rate.
        num_frame = video.split('.')[0].split('_')[4]
        print("Num frame = " + str(num_frame))

        # Calcola la grandezza delle sequenze.
        frame_contigui = int(num_frame) * CONTIGUOUS_SECONDS
        print("Grandezza sequenze = " + str(frame_contigui))

        # Apre il video.
        cap = cv2.VideoCapture(path_ + "/" + videoFile)

        # Contiene array di sequenze da 5 secondi ciascuno.
        contiguous_sequences = []

        # Contiene sequenze continue di frame di lunghezza massima 5 secondi.
        contiguous_frames = []

        while cap.isOpened():
            print(str(len(contiguous_frames)))

            # Quando la sequenza raggiunge la grandezza massima la aggiunge
            # all'array e va avanti cercando altre sequenze.
            if len(contiguous_frames) == frame_contigui:
                print("Trovata sequenza di " + str(CONTIGUOUS_SECONDS) + " secondi")

                # Aggiunge la sequenza trova all'array.
                contiguous_sequences.append(contiguous_frames)

                print("Sono presenti " + str(len(contiguous_sequences)) + " sequenze nell'array")

                # Svuota l'array per cercare nuove sequenze.
                contiguous_frames = []

            # Apre il frame.
            ret, image_o = cap.read()

            # Cambia la dimensione del frame in modo che ogni video abbia la stessa dimensione.
            image = cv2.resize(image_o, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

            if not ret:
                break

            # Trova il viso nell'immagine.
            rects = detector(image, 1)

            if len(rects) == 1:
                print("Aggiunto frame")
                contiguous_frames.append(image)

            # Se il frame non contiene un viso allora cancella l'intera sequenza.
            else:
                print("Frame senza un soggetto visibile, sequenza cancellata")
                contiguous_frames = []

            # Ha raggiunto il numero massimo di sequenze.
            if len(contiguous_sequences) >= (MAX_PER_VIDEO * 3):
                print("Prodotti " + str(MAX_PER_VIDEO) + " video da questo, proseguo col prossimo")
                break

        # Chiama il metodo save_video e gli passa l'array di sequenze trovate.
        save_video(contiguous_sequences, where_to_save, num_frame, video.split('.')[0])


for videoFile in os.listdir(path):
    print(path)
    create_videos(path, videoFile, destinationPath)
