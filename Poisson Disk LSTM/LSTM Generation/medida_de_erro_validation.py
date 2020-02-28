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
from sklearn.metrics import mean_squared_error


def generate():
    """ Predict particles """
    high = 7
    look_back = 15

    # get squences of particle generation test
    particlesx_test, particlesy_test = get_sequences_test()

    #Get input and [output1,output2] = [x,y] formated for keras
    network_input_test, normalized_input_test , network_output1_test, network_output2_test = prepare_sequences(particlesx_test, particlesy_test,high, look_back)
    #Create LSTM Model
    model = create_network(network_input_test, high, look_back)

    errorx, errory = error_particles(model, network_input_test, network_output1_test,network_output2_test, high)
    print(errorx)
    print(errory)


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
    print (network_input.shape[1])

    print (network_input.shape[2])
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


def error_particles(model, network_input_test, network_output1_test,network_output2_test, high):
    """ Generate particles from the neural network based on a sequence of particles """

    prediction_output1 = []
    prediction_output2 = []

    for i in range(0,len(network_input_test)):
        prediction_input = numpy.reshape(network_input_test[i], (1, 2, network_input_test.shape[2]))
        prediction_input = prediction_input / high
        prediction1, prediction2 = model.predict(prediction_input, verbose=0)
        result1 = float(prediction1)
        print (result1)
        print(network_output1_test[i])
        prediction_output1.append(result1)
        result2 = float(prediction2)
        prediction_output2.append(result2)

    errorx = mean_squared_error(network_output1_test, prediction_output1)
    errory = mean_squared_error(network_output2_test, prediction_output2)
    return errorx, errory



if __name__ == '__main__':
    generate()
