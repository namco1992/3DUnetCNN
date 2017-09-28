import os
import math

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau, TensorBoard, EarlyStopping

from utils import pickle_dump
# from model_3d import dice_coef, dice_coef_loss

K.set_image_dim_ordering('tf')


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):
    callbacks = [
        # ModelCheckpoint(model_file, save_best_only=True),
        CSVLogger(os.path.join(logging_dir, "training.log")),
        SaveLossHistory(),
        # LearningRateScheduler(partial(step_decay,
        #                       initial_lrate=initial_learning_rate,
        #                       drop=learning_rate_drop,
        #                       epochs_drop=learning_rate_epochs)),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
        TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ModelCheckpoint(
            model_file, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)]

    return callbacks


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):
    """
    Train a Keras model.
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return:
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        pickle_safe=True,
                        callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs))
