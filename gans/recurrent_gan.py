#----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
#------------------------------------------------------------------------------------------------------
import numpy as np
from keras.layers import Input, GaussianNoise, TimeDistributed, Dense, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from complexnet.layers.complex_recurrent import CLSTM
from data_processing.data_processing import get_data

class recurrent_gan():
    def __init__(self, input_dim):
        # Note, we are multiplying by 2 to reflect how complex numbers are represented
        self.input_dim = input_dim * 2

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build generator
        self.generator = self.build_generator()

        # Generator takes noise as input and generates songs
        z = Input(shape=(None, self.input_dim))
        song = self.generator(z)

        # For the combined model freeze the discriminator weights
        self.discriminator.trainable = False

        validity = self.discriminator(song)

        # Create the combined model
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(CLSTM(512, return_sequences=True))
        model.add(CLSTM(512, return_sequences=True))
        model.add(CLSTM(512, return_sequences=True))

        model.summary()

        noise = Input(shape=(None, self.input_dim))
        song = model(noise)

        return Model(noise, song)

    def build_discriminator(self):

        model = Sequential()

        model.add(CLSTM(512, return_sequences=True))
        model.add(CLSTM(512, return_sequences=True))
        model.add(LSTM(1))

        model.summary()

        song = Input(shape=(None, self.input_dim))
        validity = model(song)

        return Model(song, validity)

    def train(self, epochs):
        X_train = get_data()

        # Ground truths
        valid = np.ones((1, 1))
        valid_gen = np.ones((2,1))
        fake = np.zeros((1, 1))

        for epoch in range(epochs):
            d_loss_real = 0
            d_loss_fake = 0
            g_loss = 0
            for song in X_train:
                # Generate the new song
                song_length = np.shape(song)[1]
                noise = np.random.normal(0, 1, (1, song_length, self.input_dim))
                gen_song = self.generator.predict(noise)

                # Train discriminator
                d_loss_real += self.discriminator.train_on_batch(song, valid)
                d_loss_fake += self.discriminator.train_on_batch(gen_song, fake)


                # Train the generator
                # Note, since we are training the discriminator with 2 examples,
                #    we will do the same with the generator
                noise = np.random.normal(0, 1, (2, song_length, self.input_dim))

                g_loss += self.combined.train_on_batch(noise, valid_gen)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
