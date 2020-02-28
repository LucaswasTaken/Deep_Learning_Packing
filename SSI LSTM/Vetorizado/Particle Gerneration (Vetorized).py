""" This module generates particles by classification on 8x8 table using classification """
import pickle
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

def generate():
    porosity = input("especificar porosidade (minimo de 30%)")
    length = input("especificar lado do quadrado")
    n_cells = int(int(length)/8)
    particles_t = []
    n_generate = int((1-int(porosity)/100)*25)
    n_cat = 64
    n_part = 15
    n_ex = 10000
    particles = get_sequences(n_part, n_ex)
    network_input, normalized_input = prepare_sequences(particles, n_part, n_ex, n_cat)
    model = create_network(normalized_input, n_cat)
    for i in range(0,n_cells*n_cells):
        prediction_output = generate_particles(model, network_input, n_cat, n_generate)
        particles_t.append(prediction_output)
    create_ofs(particles_t,n_cells, n_generate,n_cat)

def get_sequences(n_part, n_ex):
    """ Get all the particles from a file """
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
    # map between notes and integers and back
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
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_cat)

    return (network_input, normalized_input)

def create_network(network_input, n_cat):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512,
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
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Load the weights to each node
    model.load_weights('weights_final.hdf5')

    return model

def generate_particles(model, network_input, n_cat, n_seq_generated):
    """ Generate particles from the neural network based on a sequence of particles """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    # generate particles
    for n_p_g in range(n_seq_generated):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_cat)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int(index)
        print (result)
        prediction_output.append(result)

        pattern = numpy.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_ofs(particles,n_cells, n_generate,n_cat):

    arq2 = open("LSTM_Particle_Classification.ofs","w")
    escritaantes = "%DEM.MATERIAL.COLOR\n1\n1 0.725 0.478 0.341 1.000\n%DEM.PARTICLE\n"+str(n_cells*n_cells*n_generate)+"\n%DEM.PARTICLE.CIRCLE\n"+str(n_cells*n_cells*n_generate)+"\n"
    arq2.write(escritaantes)
    count = 0;
    for i in range(0,n_cells*n_cells):
        for j in range(0,len(particles[i])):
            count = count+1
            x = str(particles[i][j]%int(math.sqrt(n_cat))+8*int(i%n_cells))
            y = str(int(particles[i][j]/int(math.sqrt(n_cat)))+8*int(i/n_cells))
            ident = str(count)
            escrita = ident+" 1 1 "+x+" "+y+" 0.00\n"
            arq2.write(escrita)

    arq2.close()





if __name__ == '__main__':
    generate()
