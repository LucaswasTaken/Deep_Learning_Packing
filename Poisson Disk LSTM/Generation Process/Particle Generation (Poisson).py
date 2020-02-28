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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import imutils
import cv2
import numpy as np

def generate():

    """ Predict particles """
    porosity = input("especificar porosidade (minimo de 30%)")
    length = input("especificar lado do quadrado")

    n_cells = int(int(length) / 7)
    particles_t_x = []
    particles_t_y = []
    n_generated = int((1 - int(porosity) / 100) * 49)
    high = 7
    look_back = 15
    number_total =0


    # get squences of particle generation predict
    particlesx, particlesy = get_sequences()

    #Get input and [output1,output2] = [x,y] formated for keras
    network_input, normalized_input , network_output1, network_output2 = prepare_sequences(particlesx, particlesy,high, look_back)

    #Create LSTM Model
    model = create_network(network_input, high, look_back)

    # Create Convolutional Stop Model
    model_stop =  create_network_stop()

    #Generate Particles
    for i in range(0,n_cells*n_cells):
        prediction_output1, prediction_output2 = generate_particles(model, model_stop, normalized_input, high, n_generated)
        number_total = number_total + len(prediction_output1)
        particles_t_x.append(prediction_output1)
        particles_t_y.append(prediction_output2)

    create_ofs(particles_t_x,particles_t_y,n_cells, n_generated,high,number_total)


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

        if len(particlesx[i]) > look_back:
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


def create_network_stop():

    img_width, img_height = 150, 150
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.load_weights('stop_poisson-acc_0.897_err_0.2416.hdf5')

    return model


def generate_particles(model, model_stop, network_input, high, n_seq_generated):
    """ Generate particles from the neural network based on a sequence of particles """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, network_input.shape[0]-1)
    pattern = network_input[start]
    prediction_output1 = []
    prediction_output2 = []

    for i in range (len(pattern[0])):
        prediction_output1.append(pattern[0][i]*high)
        prediction_output2.append(pattern[1][i]*high)

    count = len(pattern[0])-1

    for p_index in range(n_seq_generated):

        #Geracao das particulas
        prediction_input = numpy.reshape(pattern, (1, 2, network_input.shape[2]))
        prediction1, prediction2 = model.predict(prediction_input, verbose=0)
        prediction_output1.append(prediction1)
        prediction_output2.append(prediction2)

        patternaux1 = numpy.append(pattern[0][1:len(pattern[0]-1)],prediction1/high)
        patternaux2 = numpy.append(pattern[1][1:len(pattern[1]-1)], prediction2/high)
        pattern[0] = patternaux1
        pattern[1] = patternaux2
        count = count+1;


        # Criterio de parada
        if count > 20:
            print(count)
            img = np.zeros([7000, 7000, 3], dtype=np.uint8)
            for j in range(0, len(prediction_output1)):
                x = int(prediction_output1[j]*1000)
                y = int(prediction_output2[j]*1000)
                cv2.circle(img, (x, y), 500, (0, 0, 255), -1)
            img = cv2.resize(img, (150, 150))
            img = np.reshape(img, [1, 150, 150, 3])
            classes = model_stop.predict(img)
            if int(classes) == 1:
                return prediction_output1, prediction_output2


    return prediction_output1, prediction_output2


def create_ofs(particlesx,particlesy,n_cells, n_generated,n_cat,number_total):

    arq2 = open("LSTM_Particle_Poisson.ofs", "w")
    escritaantes = "%DEM.MATERIAL.COLOR\n1\n1 0.725 0.478 0.341 1.000\n%DEM.PARTICLE\n" + str(number_total) + "\n%DEM.PARTICLE.CIRCLE\n" + str(number_total) + "\n"
    arq2.write(escritaantes)
    count = 0;
    for i in range(0, n_cells * n_cells):
        for j in range(0, len(particlesx[i])):
            count = count + 1
            x = str(float(particlesx[i][j])+int((i/n_cells))*n_cat)
            y = str(float(particlesy[i][j])+(i%n_cells)*n_cat)
            ident = str(count)
            escrita = ident + " 1 0.5 " + x + " " + y + " 0.00\n"
            arq2.write(escrita)

    arq2.close()





if __name__ == '__main__':
    generate()
