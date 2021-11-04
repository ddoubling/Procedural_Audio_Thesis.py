from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

from autoencoder import Autoencoder
from vae import vae


LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10


def load_mnist():
    """Loader of MNSIST Dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def select_images(images, labels, num_images=10):
    """Image selecter for reconstruction and generation, using the name indexes to match label"""
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
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
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 128),
        conv_kernels=(3, 3, 3, 3 ),
        conv_strides=(1, 2, 2, 1),
        latent_space=2
    )
    autoencoder.archetecturePrint()
    autoencoder.compiler(learning_rate)
    autoencoder.train(x_train, x_test,batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    autoencoder = train(x_train, x_test, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")

    autoencoder = Autoencoder.load("model")


    num_sample_images_to_show = 10
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)