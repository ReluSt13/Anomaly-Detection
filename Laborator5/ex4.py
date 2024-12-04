import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

noise_factor = 0.35
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, 0.0, 1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0.0, 1.0)

class ConvAutoencoder(models.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same', input_shape=(28, 28, 1)),
            layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
        ])
        self.decoder = models.Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test))

reconstructions = autoencoder.predict(x_train)
train_loss = tf.keras.losses.mse(x_train, reconstructions)
threshold = np.mean(train_loss) + np.std(train_loss)

reconstructions_test = autoencoder.predict(x_test)
test_loss = tf.keras.losses.mse(x_test, reconstructions_test)
test_classification = test_loss < threshold

reconstructions_test_noisy = autoencoder.predict(x_test_noisy)
test_loss_noisy = tf.keras.losses.mse(x_test_noisy, reconstructions_test_noisy)
test_classification_noisy = test_loss_noisy < threshold

accuracy_original = np.mean(test_classification)
accuracy_noisy = np.mean(test_classification_noisy)

print(f'Accuracy on original test images: {accuracy_original}')
print(f'Accuracy on noisy test images: {accuracy_noisy}')

def plot_results(x_test, x_test_noisy, reconstructions, reconstructions_noisy, graph_nr):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(4, 5, i + 1)
        plt.title('Original')
        plt.imshow(x_test[i], cmap='gray')
        plt.axis('off')

        plt.subplot(4, 5, i + 6)
        plt.title('Noisy')
        plt.imshow(x_test_noisy[i], cmap='gray')
        plt.axis('off')

        plt.subplot(4, 5, i + 11)
        plt.title('Reconstructed')
        plt.imshow(reconstructions[i], cmap='gray')
        plt.axis('off')

        plt.subplot(4, 5, i + 16)
        plt.title('Reconstr noisy')
        plt.imshow(reconstructions_noisy[i], cmap='gray')
        plt.axis('off')
    plt.savefig(f'ex4_{graph_nr}.pdf')
    plt.show()

plot_results(x_test, x_test_noisy, reconstructions_test, reconstructions_test_noisy, '1')

history_denoising = autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=64, validation_data=(x_test_noisy, x_test))

reconstructions_denoising = autoencoder.predict(x_test)
reconstructions_denoising_noisy = autoencoder.predict(x_test_noisy)
plot_results(x_test, x_test_noisy, reconstructions_denoising, reconstructions_denoising_noisy, '2')