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
    """ Predict particles """
    high = 7
    look_back = 15
    n_seq_generated = 15
    # get squences of particle generation predict
    particlesx, particlesy = get_sequences()

    # get squences of particle generation test
    particlesx_test, particlesy_test = get_sequences_test()

    #Get input and [output1,output2] = [x,y] formated for keras
    network_input, normalized_input , network_output1, network_output2 = prepare_sequences(particlesx, particlesy,high, look_back)

    #Create LSTM Model
    model = create_network(network_input, high, look_back)
    prediction_output1, prediction_output2 = generate_particles(model, network_input, high, n_seq_generated)
    create_ofs(prediction_output1,prediction_output2, n_seq_generated,high)


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

def get_sequences_test():
    """ Get all particle examples from traindata.txt """
    particlesx = []
    particlesy = []

    arqx = open("poissontestdatax.txt", "r")
    arqy = open("poissontestdatax.txt", "r")
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
    # normalize input
    normalized_input = network_input / high

    return network_input, normalized_input, network_output1, network_output2

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
    model = Model(input=inputs, output=[output1, output2])
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.load_weights('weights-improvement-0.06-full.hdf5')
    return model

def generate_particles(model, network_input, high, n_seq_generated):
    """ Generate particles from the neural network based on a sequence of particles """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, network_input.shape[2]-1)

    pattern = network_input[start]
    prediction_output1 = []
    prediction_output2 = []

    for p_index in range(n_seq_generated):
        prediction_input = numpy.reshape(pattern, (1, 2, network_input.shape[2]))
        prediction_input = prediction_input

        prediction1, prediction2 = model.predict(prediction_input, verbose=0)
        print(prediction1)
        prediction_output1.append(prediction1)
        prediction_output2.append(prediction2)

        pattern = numpy.append(pattern,prediction1)
        pattern = numpy.append(pattern, index2)
        pattern = pattern[1:len(pattern)-1]

    return prediction_output1, prediction_output2


def create_ofs(prediction_output1,prediction_output2, n_seq_generated,high):
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
