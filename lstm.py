""" This module prepares midi file data and feeds it to the neural
    network for training """

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
import numpy
from keras.utils import np_utils
import glob
import music21
from music21 import converter, instrument, note, chord, stream
import pickle


def extract_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("/home/ml_new/mihika/midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('/home/ml_new/mihika/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def get_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitch_name = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_name))

    input_network = []
    output_network = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_network.append([note_to_int[char] for char in sequence_in])
        output_network.append(note_to_int[sequence_out])

    n_patterns = len(input_network)

    # reshape the input into a format compatible with LSTM layers
    input_network = numpy.reshape(input_network, (n_patterns, sequence_length, 1))
    # normalize input
    input_network = input_network / float(n_vocab)

    output_network = np_utils.to_categorical(output_network)

    return input_network, output_network


def create_network(input_network, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(input_network.shape[1], input_network.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, input_network, output_network):
    """ train the neural network """
    filepath = "/home/ml_new/mihika/weights.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(input_network, output_network, epochs=50, batch_size=64, callbacks=callbacks_list)


def train_network():
    """ Train a Neural Network to generate music """

    notes = extract_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    input_network, output_network = get_sequences(notes, n_vocab)

    model = create_network(input_network, n_vocab)

    train(model, input_network, output_network)


if __name__ == '__main__':
    train_network()
