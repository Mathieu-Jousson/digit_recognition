import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from model import create_cnn_model
from utils import recentrer_et_redimensionner
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_and_save():
    print("--- Chargement des données MNIST ---")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

    print("--- Recadrage et Standardisation des images ---")
    x_train = np.array([recentrer_et_redimensionner(img) for img in x_train_raw])
    x_test = np.array([recentrer_et_redimensionner(img) for img in x_test_raw])

    # Format 4D et normalisation
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train_raw, 10)
    y_test = tf.keras.utils.to_categorical(y_test_raw, 10)

    print("--- Entraînement du modèle ---")
    model = create_cnn_model()
    model.fit(x_train, y_train, batch_size=200, epochs=10, validation_data=(x_test, y_test))

    print("--- Sauvegarde du modèle ---")
    model_name="cnn_model_mnist.keras"
    model.save(model_name)
    print(f"Modèle sauvegardé sous {model_name}")

if __name__ == "__main__":
    train_and_save()