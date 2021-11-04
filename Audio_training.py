import os
import sys
from matplotlib import pyplot as plt
import numpy as np

from soundgenerator import SoundGenerator
from vae import vae, RECONSTRUCTIONLOSS
from autoencoder import Autoencoder
from train_preprocess import FRAME_SIZE, HOP_LENGTH, DURATION, SAMPLE_RATE, MONO
from datetime import datetime
now = datetime.now()



LEARNING_RATE = 0.0005


BATCH_SIZE = 128
EPOCHS = 35

TEST_SPEC_PATH = r"C:\Users\dwill\Desktop\drum\test_spec"
TRAIN_SPEC_PATH = r"C:\Users\dwill\Desktop\drum\train_spec"

def load_train(spectrograms_path):
    """loader of spectrograms for training dataset"""
    dataset = []
    names = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # spectrogram dimensions equal: n_bins, n_frames, 1 channel for greyscale
            dataset.append(spectrogram)
            names.append(file_name)
            #print(names)
    dataset = np.array(dataset)
    dataset = dataset[
        ..., np.newaxis]  # x_train array becomes -> [(samples in dataset], 256(n_bins), 64 (n_frames, 1 (greyscale)]

    return dataset

def load_test(spectrograms_path):
    """loader of spectrograms for test dataset"""
    augmented = []
    names = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # spectrogram dimensions equal: n_bins, n_frames, 1 channel for greyscale
            augmented.append(spectrogram)
            names.append(file_name)
    dataset = np.array(augmented)
    dataset = dataset[
        ..., np.newaxis]  # x_train array becomes -> [(samples in dataset], 256(n_bins), 64 (n_frames, 1 (greyscale)]

    return dataset, names

def select_images(images, labels, num_images):
    """Image selecter for reconstruction and generation, using the name indexes to match instrument type"""
    sample_labels = []
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    #print(sample_images)
    #print(sample_images_index)
    for index in sample_images_index:
        if "Snare" in labels[index]:
            sample_labels.append(0)
        elif "Tom" in labels[index]:
            sample_labels.append(1)
        elif "Crash" in labels[index]:
            sample_labels.append(2)
        elif "Kick" in labels[index]:
            sample_labels.append(3)
        elif "HiHat" in labels[index]:
            sample_labels.append(4)
        elif "Ride" in labels[index]:
            sample_labels.append(5)
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    """Image reconstruction and generation in side by side comparisons"""
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.savefig("reconstruction.png")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """latent space representation, showing classification across the latent space. Only effective in two dimensions"""
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.savefig("latent_space.png")
    plt.show()


def train(x_train, x_test, learning_rate, batch_size, epochs):
    """Network archetecture setup and control over autencoder or vae calling"""
    autoencoder = vae(
        input_shape=(256, 64, 1),
        conv_filters=(32, 64, 128, 256, 512),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space=100
    )
    autoencoder.archetecturePrint()
    autoencoder.compiler(learning_rate)
    autoencoder.train(x_train, x_test, batch_size, epochs)
    return autoencoder

def architecturesetup():
    print("Training Parameters \n","Learning Rate:",LEARNING_RATE, " Batch Size:", BATCH_SIZE, " Epochs:", EPOCHS, "\n")

def autoencodersetup():
    print("Autoencoder Setup\n", "Input shape:", autoencoder.inputShape, "\n Convolutional Filters:",
          autoencoder.convFilters, "\n Convolutional Kernels:", autoencoder.convKernels, "\n Convolutional Strides:",
          autoencoder.convStrides, "\n Latent Space Dimensions:", autoencoder.latentSpace,
          "\n Reconstruction Loss Weight:", RECONSTRUCTIONLOSS, "\n")

def preprocesssetup():
    print("Preprocess Parameters of Audio \n", "Frame Size:", FRAME_SIZE, "\n Hop Length:", HOP_LENGTH,
          "\n Duration in Seconds:", DURATION, "\n Sample Rate Hz:", SAMPLE_RATE, "\n Mono:", MONO, "\n")

if __name__ == "__main__":
    X_train = load_train(TRAIN_SPEC_PATH)
    X_Test, names = load_test(TEST_SPEC_PATH)
    num = len(X_Test)

    autoencoder = train(X_train, X_Test, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    filename = open(os.path.join('model', 'architecture.txt'), 'w')
    sys.stdout = filename
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S \n")
    print("date and time of running :", dt_string)
    autoencoder.archetecturePrint()
    preprocesssetup()
    architecturesetup()
    autoencodersetup()

    sys.stdout.close()


    autoencoder = vae.load("model")
    num_sample_images_to_show = 8
    sample_images, labels = select_images(X_Test, names, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_images = num
    sample_images, sample_labels = select_images(X_Test, names, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)

    vae = vae.load(r"C:\Users\dwill\PycharmProjects\NN\model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

