""" This module train a LSTM network to generate particles """

import numpy
import math
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error


def train_network():
    """ Train a Neural Network to generate particles """

    high = 7
    look_back = 15
    # get squences of particle generation
    particlesx, particlesy = get_sequences()
    #Get input and [output1,output2] = [x,y] formated for keras
    network_input, network_output1, network_output2 = prepare_sequences(particlesx, particlesy,high, look_back)

    #Create LSTM Model
    model = create_network(network_input, high, look_back)

    #Train model
    train(model, network_input, network_output1, network_output2)


def get_sequences():
    """ Get all particle examples from traindata.txt """
    particlesx = []
    particlesy = []

    arqx = open("poissontraindatax.txt", "r")
    arqy = open("poissontraindatax.txt", "r")
    for i in arqx:
        readx = arqx.readline().split()
        ready = arqy.readline().split()
        auxx =[]
        auxy=[]
        for j in range(0,len(readx)):
            auxx.append(float(readx[j]))
            auxy.append(float(ready[j]))
        particlesx.append(auxx)
        particlesy.append(auxy)
    arqx.close()
    arqy.close()

    return particlesx, particlesy


def prepare_sequences(particlesx, particlesy,high, look_back):

    """ Prepare the sequences used by the Neural Network """
    network_input = []
    network_output1 = []
    network_output2 = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(particlesx), 1):
        if(len(particlesx[i])>look_back):
            sequence_inx = particlesx[i][0:look_back]
            sequence_iny = particlesy[i][0:look_back]
            sequence_in = [sequence_inx,sequence_iny]
            sequence_out1 = particlesx[i][look_back]
            sequence_out2 = particlesy[i][look_back]
            network_input.append(sequence_in)
            network_output1.append(sequence_out1)
            network_output2.append(sequence_out2)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers

    network_input = numpy.reshape(network_input, (n_patterns, 2 , look_back))
    network_output1 = numpy.array(network_output1)
    network_output2 = numpy.array(network_output2)
    print (network_input.shape[1])

    print (network_input.shape[2])
    # normalize input
    network_input = network_input / high

    return network_input, network_output1, network_output2

def create_network(network_input,high, look_back):
    """ create the structure of the neural network """
    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))
    x = LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True)(inputs)
    x = (Dropout(0.4)(x))
    x = (LSTM(1024, return_sequences=True)(x))
    x = (Dropout(0.4)(x))
    x = (LSTM(2048, return_sequences=True)(x))
    x = (Dropout(0.4)(x))
    x = (LSTM(1024, return_sequences=True)(x))
    x = (Dense(512)(x))
    x = (LSTM(512)(x))
    x = (Dense(256)(x))
    output1 = Dense(1)(x)
    output2 = (Dense(1)(x))
    model = Model(input=inputs, output=[output1,output2])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.load_weights('weights-improvement.hdf5')
    return model


def train(model, network_input, network_output1, network_output2):
    """ train the neural network """
    filepath = "weights-improvement.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, [network_output1, network_output2], epochs=1000, batch_size=120, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
