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
from sklearn.model_selection import train_test_split

def train_network():
    """ Train a Neural Network to generate particles """
    
    #number of spaces in the domain (8x8)
    n_cat = 64
    
    #number of particles generated in the domain
    n_part = 15

    #number of trainning examples (train+test)
    n_ex = 18000

    # get squences of particle generation
    particlesx, particlesy = get_sequences(n_part, n_ex, n_cat)

    #Get input and [output1,output2] = [x,y] formated for keras
    network_input, network_output1, network_output2 = prepare_sequences(particlesx, particlesy, n_part, n_ex, n_cat)

    #Create LSTM Model
    model = create_network(network_input, n_cat)

    #Train model
    train(model, network_input, network_output1, network_output2)


def get_sequences(n_part, n_ex, n_cat):
    """ Get all particle examples from traindata.txt """
    s = (n_ex,n_part)
    particlesx = numpy.zeros(s, dtype=int)
    particlesy = numpy.zeros(s, dtype=int)

    arq = open("traindata.txt", "r")

    for i in range(0,n_ex):
        aux = arq.readline().split()
        for j in range(0,n_part):
            particlesx[i][j] = int(int(aux[j]) % int(math.sqrt(n_cat)))
            particlesy[i][j] = int(int(aux[j]) / int(math.sqrt(n_cat)))
    arq.close()

    return particlesx, particlesy


def prepare_sequences(particlesx, particlesy, n_part, n_ex, n_cat):

    """ Prepare the sequences used by the Neural Network """
    sequence_length = int(n_part-1)
    network_input = []
    network_output1 = []
    network_output2 = []

    # create input sequences and the corresponding outputs
    for i in range(0, n_ex, 1):
        sequence_inx = particlesx[i][0:sequence_length]
        sequence_iny = particlesy[i][0:sequence_length]
        sequence_in = [sequence_inx,sequence_iny]
        sequence_out1 = particlesx[i][sequence_length]
        sequence_out2 = particlesy[i][sequence_length]
        network_input.append(sequence_in)
        network_output1.append(sequence_out1)
        network_output2.append(sequence_out2)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    
    network_input = numpy.reshape(network_input, (n_patterns, 2 , sequence_length))
    print (network_input.shape[1])

    print (network_input.shape[2])
    # normalize input
    network_input = network_input / float(math.sqrt(n_cat))
    network_output1 = np_utils.to_categorical(network_output1)
    network_output2 = np_utils.to_categorical(network_output2)


    return network_input, network_output1, network_output2

def create_network(network_input, n_cat):
    """ create the structure of the neural network """
    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))
    x = LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True)(inputs)
    x = (Dropout(0.3)(x))
    x = (LSTM(512, return_sequences=True)(x))
    x = (Dropout(0.3)(x))
    x = (Dense(256)(x))
    x = (LSTM(512)(x))
    x = (Dense(256)(x))
    x = (Dropout(0.3)(x))
    output1 = Dense(int(math.sqrt(n_cat)))(x)
    output1 = (Activation('softmax')(output1))
    output2 = (Dense(int(math.sqrt(n_cat)))(x))
    output2 = (Activation('softmax')(output2))
    model = Model(input=inputs, output=[output1,output2])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output1, network_output2):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
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
