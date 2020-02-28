""" This module train a LSTM network for particle generation using a vetorized strategy and classification """

import numpy
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

    #numero de casas no tabuleiro (8x8)
    n_cat = 64
    #numero de particulas a serem geradas
    n_part = 15
    #numero de exemplos a serem lidos
    n_ex = 18000

    # get squences of particle generation
    particles = get_sequences(n_part, n_ex)

    input_train, input_test, output_train, output_test = prepare_sequences(particles, n_part, n_ex, n_cat)

    model = create_network(input_train, n_cat)

    train(model, input_train, input_test, output_train, output_test)

def get_sequences(n_part, n_ex):
    """ Get all the particles position from the txt file"""
    s = (n_ex,n_part)
    particles = numpy.zeros(s,dtype=int)
    arq = open("traindata.txt","r")

    for i in range(0,n_ex):
        
        aux = arq.readline().split()
        for j in range(0,n_part):
            particles[i][j] = int(aux[j])
    arq.close()

    return particles

def prepare_sequences(particles, n_part, n_ex, n_cat):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = int(n_part-1)

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, n_ex, 1):
        sequence_in = particles[i][0:sequence_length]
        sequence_out = particles[i][sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_cat)

    network_output = np_utils.to_categorical(network_output)

    seed = 7
    numpy.random.seed(seed)
    input_train, input_test, output_train, output_test = train_test_split(network_input, network_output, test_size=0.3, random_state=seed)
    return (input_train, input_test, output_train, output_test)

def create_network(network_input, n_cat):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_cat))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, input_train, input_test, output_train, output_test):
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

    model.fit(input_train, output_train,validation_data=(input_test,output_test), epochs=500, batch_size=2000, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
