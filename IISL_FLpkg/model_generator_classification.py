import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import pdb

from .model_custom_classification import CustomModelList_Classification, CustomModel_Classification


def model_generation(N, input_size, code):
    random_seed = 3
    tf.random.set_seed(random_seed)
    all_models = CustomModelList_Classification()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    # Occupancy Estimation
    if code == 1:
        for i in range(N):
            model = models.Sequential()
            model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(4, activation='softmax'))
            tf.random.set_seed(random_seed)
            model1 = CustomModel_Classification(model)
            model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
            all_models.append(model1)
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))
        tf.random.set_seed(random_seed)
        central_server = CustomModel_Classification(model)
        central_server.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
        return all_models, central_server
    # MNIST
    elif code == 2:
        for i in range(N):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 1)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='softmax'))
            tf.random.set_seed(random_seed)
            model1 = CustomModel_Classification(model)
            model1.compile(optimizer='adam', loss=loss_fn)
            all_models.append(model1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        tf.random.set_seed(random_seed)
        central_server = CustomModel_Classification(model)
        central_server.compile(optimizer='adam', loss=loss_fn)

        return all_models, central_server

    # Movement
    elif code == 3:
        for i in range(N):
            model = models.Sequential()
            model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(2, activation='softmax'))
            tf.random.set_seed(random_seed)
            model1 = CustomModel_Classification(model)
            model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
            all_models.append(model1)
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))
        tf.random.set_seed(random_seed)
        central_server = CustomModel_Classification(model)
        central_server.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
        return all_models, central_server

    # CTG
    elif code == 4:
        for i in range(N):
            model = models.Sequential()
            model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(4, activation='softmax'))
            tf.random.set_seed(random_seed)
            model1 = CustomModel_Classification(model)
            model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
            all_models.append(model1)
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))
        tf.random.set_seed(random_seed)
        central_server = CustomModel_Classification(model)
        central_server.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
        return all_models, central_server



# def model_generation(N, metric):
#     random_seed = 3
#     tf.random.set_seed(random_seed)
  
#     all_models = CustomModelList()
#     loss_fn = keras.losses.SparseCategoricalCrossentropy()
#     for i in range(N):
#         model = models.Sequential()
#         model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
#         model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#         model.add(layers.MaxPooling2D((2, 2)))
#         model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(layers.MaxPooling2D((2, 2)))
#         model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(layers.MaxPooling2D((2, 2)))
#         model.add(layers.Flatten())
#         model.add(layers.Dense(10, activation='softmax'))
#         tf.random.set_seed(random_seed)
#         model1 = CustomModel(model)
#         model1.compile(optimizer='sgd', loss=loss_fn, metrics=metric)
#         all_models.append(model1)
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation='softmax'))
#     tf.random.set_seed(random_seed)
#     central_server = CustomModel(model)
#     central_server.compile(optimizer='sgd', loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#     return all_models, central_server


# # Occupancy
# def model_generation(N, input_size):
#     random_seed = 3
#     tf.random.set_seed(random_seed)
  
#     all_models = CustomModelList_Classification()
#     loss_fn = keras.losses.SparseCategoricalCrossentropy()
#     for i in range(N):
#         model = models.Sequential()
#         model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
#         model.add(layers.Dense(32, activation='relu'))
#         model.add(layers.Dense(4, activation='softmax'))
#         tf.random.set_seed(random_seed)
#         model1 = CustomModel_Classification(model)
#         model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
#         all_models.append(model1)
#     model = models.Sequential()
#     model.add(layers.Dense(32, activation='relu', input_shape=(None, 1, input_size)))
#     model.add(layers.Dense(32, activation='relu'))
#     model.add(layers.Dense(4, activation='softmax'))
#     tf.random.set_seed(random_seed)
#     central_server = CustomModel_Classification(model)
#     central_server.compile(optimizer='adam', loss=loss_fn)
#     return all_models, central_server
