from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from . import data_processor


def train(x, y):
    x = data_processor.transform_x(x)
    """
    array([[[ 1, 11],
        [ 2, 12],
        [ 3, 13]],
       [[ 2, 12],
        [ 3, 13],
        [ 4, 14]],
       [[ 3, 13],
        [ 4, 14],
        [ 5, 15]],
       [[ 4, 14],
        [ 5, 15],
        [ 6, 16]],
       [[ 5, 15],
        [ 6, 16],
        [ 7, 17]]])
        """
    model = Sequential()
    model.add(LSTM(3, input_shape=(x.shape[1], x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(8, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(x, y, verbose=1, epochs=70)
    return model


