import tensorflow as tf
from keras.layers import LSTM, GRU, Dense
from keras import regularizers
import graphviz

input_layer = tf.keras.Input(shape=(700, 66))

lstm1 = LSTM(128, return_sequences=True,
             bias_initializer="glorot_uniform")(input_layer)
gru1 = GRU(128, return_sequences=True)(lstm1)

lstm2 = LSTM(128, return_sequences=True, activity_regularizer=regularizers.l2(1e-5))(gru1)
gru2 = GRU(128, return_sequences=True)(lstm2)

lstm3 = LSTM(128, return_sequences=True, activity_regularizer=regularizers.l2(1e-5))(gru2)
gru3 = GRU(128, return_sequences=True)(lstm3)

lstm4 = LSTM(128, return_sequences=True, activity_regularizer=regularizers.l2(1e-5))(gru3)
gru4 = GRU(128, return_sequences=True)(lstm4)

lstm5 = LSTM(128, return_sequences=True, activity_regularizer=regularizers.l2(1e-5))(gru4)
gru5 = GRU(128, return_sequences=True)(lstm5)

lstm6 = LSTM(128, return_sequences=True, activity_regularizer=regularizers.l2(1e-5))(gru5)
gru6 = GRU(128, return_sequences=True)(lstm6)

lstm7 = LSTM(128)(gru6)

dense1 = Dense(32, bias_initializer="glorot_uniform")(lstm7)
dense2 = Dense(7, activation="softmax")(dense1)

model = tf.keras.Model(inputs=input_layer, outputs=dense2)
tf.keras.utils.plot_model(model, "ciao.png", show_shapes=True)