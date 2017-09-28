import os
import sys
import logging

import tables
from keras.models import load_model


HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)

from config import config
from model.generator import get_training_and_validation_generators
from model import build_model
from model.training import train_model


def main(overwrite=False):
    with tables.open_file(config["hdf5_file"], "r") as hdf5_file_opened:
        if not overwrite and os.path.exists(config["model_file"]):
            logging.info("Loading pretrained model from: {}".format(config['model_file']))
            m = load_model(config["model_file"])
        else:
            # instantiate new model
            logging.info("Train a new model...")
            m = build_model(input_shape=config["input_shape"], initial_learning_rate=config["initial_learning_rate"])

        # get training and testing generators
        train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
            hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"], overwrite=overwrite,
            validation_keys_file=config["validation_file"], training_keys_file=config["training_file"])

        # run training
        train_model(model=m, model_file=config["model_file"], training_generator=train_generator,
                    validation_generator=validation_generator, steps_per_epoch=nb_train_samples,
                    validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"],
                    learning_rate_drop=config["learning_rate_drop"],
                    learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])


if __name__ == "__main__":
    main(overwrite=False)
