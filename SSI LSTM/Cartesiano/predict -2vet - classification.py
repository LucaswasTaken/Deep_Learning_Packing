""" This module generates particles in a square domain """
import pickle
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

def generate():
    n_cat = 64
    n_part = 15
    n_ex = 3000
    n_seq_generated = 90
    particlesx, particlesy = get_sequences(n_part, n_ex, n_cat)
    network_input, normalized_input = prepare_sequences(particlesx, particlesy, n_part, n_ex, n_cat)
    model = create_network(normalized_input, n_cat)
    prediction_output1, prediction_output2 = generate_particles(model, network_input, n_cat, n_seq_generated)
    create_ofs(prediction_output1,prediction_output2, n_seq_generated,n_cat)


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

    # normalize input
    normalized_input = network_input / float(math.sqrt(n_cat))

    return network_input, normalized_input

def create_network(network_input, n_cat):
    """ create the structure of the neural network """
    inputs = Input(shape=(network_input.shape[1], network_input.shape[2]))
    x = LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True)(inputs)
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
    model = Model(input=inputs, output=[output1, output2])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('1000epochs-8.213val-improvement-828-0.0195-bigger.hdf5')
    return model

def generate_particles(model, network_input, n_cat, n_seq_generated):
    """ Generate particles from the neural network based on a sequence of particles """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, network_input.shape[2]-1)

    pattern = network_input[start]
    prediction_output1 = []
    prediction_output2 = []

    # generate 500 notes
    for note_index in range(n_seq_generated):
        prediction_input = numpy.reshape(pattern, (1, 2, network_input.shape[2]))
        prediction_input = prediction_input / float(math.sqrt(n_cat))

        prediction1, prediction2 = model.predict(prediction_input, verbose=0)

        index1 = numpy.argmax(prediction1)
        result1 = int(index1)
        print (result1)
        prediction_output1.append(result1)
        index2 = numpy.argmax(prediction2)
        result2 = int(index2)
        print (result2)
        prediction_output2.append(result2)

        pattern = numpy.append(pattern,index1)
        pattern = numpy.append(pattern, index2)
        pattern = pattern[1:len(pattern)-1]

    return prediction_output1, prediction_output2

def create_ofs(prediction_output1,prediction_output2, n_seq_generated,n_cat):
    for i in range(0,int(n_seq_generated/15)):
        name1 = "LSTM_Particle_"
        name2 = str(i)
        name3 = ".ofs"
        name = name1+name2+name3
        arq2 = open(name,"w")
        arq2.write("%DEM.MATERIAL.COLOR\n1\n1 0.725 0.478 0.341 1.000\n%DEM.PARTICLE\n15\n%DEM.PARTICLE.CIRCLE\n15\n")
        for j in range(0,15):
            x = str(prediction_output1[15*i+j])
            y = str(int(prediction_output2[15*i+j]))
            ident = str(j+1)
            escrita = ident+" 1 1 "+x+" "+y+" 0.00\n"
            arq2.write(escrita)
        arq2.close()





if __name__ == '__main__':
    generate()
