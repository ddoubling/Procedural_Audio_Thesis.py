import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, \
    Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle

tf.compat.v1.disable_eager_execution() #disables execution of calculations prior to encoding graph
RECONSTRUCTIONLOSS = 100000

class vae:
    """
    Presenting the Variational Autoencoder Architecture

    """
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space):
        """constructor"""
        self.inputShape = input_shape  # size of image shape
        self.convFilters = conv_filters  # tuple of filters
        self.convKernels = conv_kernels  # size of square kernel by layer
        self.convStrides = conv_strides  # sample size
        self.latentSpace = latent_space  # bottleneck within network
        self.reconstruction_loss_weight = RECONSTRUCTIONLOSS

        self.encoder = None
        self.decoder = None
        self.model = None

        self.numConvLayers = len(conv_filters)
        self.returnShape = None  # shape prior to flattening in bottleneck
        self.model_input = None

        self.build()




    def build(self):  # In Charge of building Autoencoder
        """caller of encoder, decoder and autoencoder"""
        self.buildEncoder()
        self.buildDecoder()
        self.buildAutoencoder()

    def buildAutoencoder(self):
        """Passing of model from encoder to decoder"""
        modelInput = self.model_input
        modelOutput = self.decoder(self.encoder(modelInput))
        self.model = Model(modelInput, modelOutput, name="Autoencoder")

    def buildEncoder(self):
        """constructon of encoder"""
        encoder_input = self.addEncoderInput()
        conv_layers = self.addConvLayers(encoder_input)
        encoder_bottleneck = self.addToLatentSpace(conv_layers)
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, encoder_bottleneck, name="encoder")

    def buildDecoder(self):
        """construction of decoder"""
        decoder_input = self.addDecoderInput()  # Create decoder input
        dense_layer = self.addDenseLayer(decoder_input)  # insert input into dense layer return
        reshape_layer = self.addReshapeLayer(dense_layer)  # return Reshape of dense layer
        conv_transpose_layers = self.addConvTransposeLayers(
            reshape_layer)  # Add Convolutional Transpose Layers to mirror Convolutional layers used in encoding
        decoder_output = self.addDecoderOutput(conv_transpose_layers)  # Take Transposed layers to output layer
        self.decoder = Model(decoder_input, decoder_output,
                             name="decoder")  # Instansiate Keras Model using decoder input and output

    def addEncoderInput(self):
        """passes input shape to encoder"""
        return Input(shape=self.inputShape, name="encoder_input")  # passing input shape to encoder

    def addDecoderInput(self):
        """passes latent space to decoder"""
        return Input(shape=self.latentSpace, name="decoderInput")

    def addDenseLayer(self, decoder_input):
        """ Reshape from latent dimension to flattened neuron size"""
        neuronNo = np.prod(self.returnShape)  # returns product of shape of flattened array
        dense_layer = Dense(neuronNo, name="decoderDense")(decoder_input)
        return dense_layer

    def addReshapeLayer(self, dense_layer):  # Return to same dimension as encoder prior to fattening
        """Reshaping dense layer after flattening"""
        reshape_layer = Reshape(self.returnShape)(dense_layer)
        return reshape_layer

    def addConvTransposeLayers(self, x):  # graph of Layers x
        """Add convolutional transpose Blocks. Reverse order looping of convolutional layers, stopping at layer 1"""
        for layerIndex in reversed(
                range(1, self.numConvLayers)):  # reversed for loop for each Convolutional Transposed layer
            x = self.addConvTransposeLayer(layerIndex, x)
        return x

    def addConvTransposeLayer(self, layerIndex, x):
        """Creation of iterative convolutional blocks within encoder"""
        layerNo = self.numConvLayers - layerIndex
        convTransposeLayer = Conv2DTranspose(
            filters=self.convFilters[layerIndex],
            kernel_size=self.convKernels[layerIndex],
            strides=self.convStrides[layerIndex],
            padding="same",
            name=f"decoderConvTransposeLayer_{layerNo}"
        )
        x = convTransposeLayer(x) # getting keras layer and applying to graph of layers
        x = ReLU(name=f"decoder_ReLu_{layerNo}")(x)
        x = BatchNormalization(name=f"decoderBN_{layerNo}")(x)
        x = Dropout((0.1), name=f"decoderDrop_{layerNo}")(x)
        return x

    def addDecoderOutput(self, x):
        """Taking final output and adding activation function"""
        convTransposeLayer = Conv2DTranspose(
            filters=1,
            kernel_size=self.convKernels[0],
            strides=self.convStrides[0], padding="same",
            name=f"decoderConvTransposeLayer_{self.numConvLayers}")
        x = convTransposeLayer(x)
        outputLayer = Activation("sigmoid", name="Output_Sigmoid_Layer")(x)
        return outputLayer

    def addConvLayers(self, encoderInput):
        """Creation of itterative convolutional blocks within encoder"""
        x = encoderInput
        for layerIndex in range(self.numConvLayers):
            x = self.addConvLayer(layerIndex, x)
        return x

    def addConvLayer(self, layerIndex, x):
        """Adding a convolutional block to a graph of layers, considting of convolutional 2d + ReLU and batch
        normalisation """

        layerNumber = layerIndex + 1
        convLayer = Conv2D(
            filters=self.convFilters[layerIndex],
            kernel_size=self.convKernels[layerIndex],
            strides=self.convStrides[layerIndex],
            padding="same",
            name=f"encoderConvLayer_{layerNumber}"
        )
        x = convLayer(x)  # getting keras layer and applying to graph of layers
        x = ReLU(name=f"encoderLayer_{layerNumber}")(x)  # apply RelU
        x = BatchNormalization(name=f"encoderBN_{layerNumber}")(x)
        x = Dropout((0.1), name=f"encoderDrop_{layerNumber}")(x)
        return x

    def addToLatentSpace(self, x):
        """Flattening of layers and pass through to latent space with gaussian sampling (dense layer)"""
        self.returnShape = K.int_shape(x)[1:] # shape prior to latent space bottleneck
        x = Flatten()(x)
        self.mu = Dense(self.latentSpace, name="mu")(x) #mean vector
        self.log_variance = Dense(self.latentSpace, name="log_Variance")(x) #variance vector

        def sampleDistrobutionPoint(args):
            """sampling a point from the normal distribution"""
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu))
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sampleDistrobutionPoint,
               name="encoder_output")([self.mu, self.log_variance])
        return x

    def reconstructionLossFunction(self, model_target, model_predicted):
        """MSE Loss Function"""
        error = model_target - model_predicted
        reconstructionLoss = K.mean(K.square(error))
        return reconstructionLoss

    def kullbackLeiblerCalculation(self, ):
        """KLD calculation to provide differences between normal distribution and standard normal distribution"""
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss

    def combinedLoss(self, model_target, model_predicted):
        """Combine Reconstruction weight * MSE + KLD"""
        reconstruction_loss = self.reconstructionLossFunction(model_target, model_predicted)
        kl_loss = self.kullbackLeiblerCalculation()
        combined_Loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_Loss

    def returnReconstructionLossWeight(self):
        """returns reconstruction weight"""
        return self.reconstruction_loss_weight


    def archetecturePrint(self):
        """summaries of network configuration"""
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compiler(self, learning_rate):
        """model compiler for autoencoder """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.combinedLoss, metrics=["accuracy"])

    def train(self, x_train, x_test, batch_size, epoch_num):
        """ train function controlling running parameters """
        history = self.model.fit(x_train,
                       x_train,
                       validation_data=(x_test,x_test),
                       batch_size=batch_size,
                       epochs=epoch_num,
                       shuffle=True)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("accuracy.png")
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("loss.png")
        plt.show()


    def save(self, save_folder="."):
        """save folder for parameters and weights"""
        self.createFolder(save_folder)  # if not existing, create save folder
        self.saveParameters(save_folder)  # save constructor parameters
        self.saveWeights(save_folder)  # save trained weights from Keras Model


    def createFolder(self, folder):
        """create folder if not already built"""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def saveParameters(self, save_folder):
        """saving of parameters for model"""
        parameters = [self.inputShape, self.convFilters, self.convKernels, self.convStrides, self.latentSpace,]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def saveWeights(self, save_folder):
        """save of model weights"""
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder):
        """loading of model weights and parameters for generation"""
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = vae(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        """load weights for generation"""
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        """returns reconstructed image and latent space assignment"""
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def encode_space(self, X):
        """predict latent space from given imput"""
        z = self.encoder.predict(X)
        plt.scatter(z[:, 0], z[:, 1], marker='o', s=0.1, c='#d53a26')
        plt.show()


if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    autoencoder = vae(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space=10

    )
    autoencoder.archetecturePrint()
