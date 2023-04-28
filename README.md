<p align="center">![logo_standard](https://user-images.githubusercontent.com/47521054/235214342-e2131ce2-b70c-4d30-a5b1-e4d558be8097.png)</p>
# Visual Language Identification through Recurrent Neural Models

Bachelor's thesis project: implementation of recurrent neural networks which analysing labial movements are able to classify the language spoken by the subject. The problem has been approached in two ways: 

- Extraction of euclidean distances located on subjects' lips from each video frame and training of Recurrent Neural Networks (LSTM and GRU) using the extracted euclidean distances. 

- Implementation of Convolutional Recurrent Neural Networks and training directly using the videos of the speaking subjects. 

Results: Final results saw the first approach as preferable, with an accuracy of 62.7%, compared with the second approach which was affected by efficiency problems. Being convolutional operations computationally complex, to run the training it was necessary to low the frames' resolution, thus affecting models capacity to classify the spoken language. 
