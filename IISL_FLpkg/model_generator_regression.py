import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from keras import layers, models

from .model_custom_regression import CustomModelList_Regression, CustomModel_Regression


def model_generation_regression(N, input_size):
    random_seed = 4
    tf.random.set_seed(random_seed)
    
    all_models = CustomModelList_Regression()
    loss_fn = keras.losses.MeanSquaredError()
    kernel_initializer = initializers.RandomNormal(stddev=0.01)
    bias_initializer=initializers.Zeros()
    for i in range(N):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(input_size,1)))
        model.add(layers.Dense(32, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        # model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dense(1))
        tf.random.set_seed(random_seed)
        model1 = CustomModel_Regression(model)
        model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)
        all_models.append(model1)
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(input_size,1)))
    model.add(layers.Dense(32, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    # model.add(layers.Dense(32, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(1))
    tf.random.set_seed(random_seed)
    central_server = CustomModel_Regression(model)
    central_server.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

    return all_models, central_server
