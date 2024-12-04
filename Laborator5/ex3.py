import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import balanced_accuracy_score

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(5, activation='relu'),
            layers.Dense(3, activation='relu')
        ])
        self.decoder = models.Sequential([
            layers.Dense(5, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(9, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_train, X_train, 
                          epochs=100, 
                          batch_size=1024, 
                          validation_data=(X_test, X_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig('ex3.pdf')
plt.show()

reconstructed_train = autoencoder.predict(X_train)
reconstructed_test = autoencoder.predict(X_test)

train_errors = np.mean(np.square(X_train - reconstructed_train), axis=1)
test_errors = np.mean(np.square(X_test - reconstructed_test), axis=1)

threshold = np.quantile(train_errors, 1 - 0.15)

train_predictions = (train_errors > threshold).astype(int)
test_predictions = (test_errors > threshold).astype(int)

train_balanced_accuracy = balanced_accuracy_score(y_train, train_predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)

print(f'Train Balanced Accuracy: {train_balanced_accuracy}')
print(f'Test Balanced Accuracy: {test_balanced_accuracy}')