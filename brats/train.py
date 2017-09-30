import os
import sys
import logging
from random import shuffle

import tables
from keras.models import load_model
from sklearn.cross_validation import KFold

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)

from config import config
from model.generator import get_training_and_validation_generators
from model import build_model
from model.training import train_model


def main(overwrite=False):
    with tables.open_file(config["hdf5_file"], "r") as hdf5_file_opened:
        nb_samples = hdf5_file_opened.root.data.shape[0]
        num_fold = 0
        kf = KFold(nb_samples, n_folds=config['n_folds'], shuffle=True, random_state=config['random_seed'])
        for train_index, valid_index in kf:
            num_fold += 1
            logging.info('Start KFold number {} from {}'.format(num_fold, config['n_folds']))
            weight_path = config['model_file'].format(num_fold)
            if not overwrite and os.path.exists(weight_path):
                logging.info("Loading pretrained model from: {}".format(weight_path))
                m = build_model(
                    input_shape=config["input_shape"], initial_learning_rate=config["initial_learning_rate"], weights=weight_path)
            else:
                # instantiate new model
                logging.info("Train a new model...")
                m = build_model(input_shape=config["input_shape"], initial_learning_rate=config["initial_learning_rate"])

            # get training and testing generators
            train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
                hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"], overwrite=overwrite,
                train_index=train_index, valid_index=valid_index)

            # run training
            train_model(model=m, model_file=weight_path, training_generator=train_generator,
                        validation_generator=validation_generator, steps_per_epoch=nb_train_samples,
                        validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"],
                        learning_rate_drop=config["learning_rate_drop"],
                        learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])


if __name__ == "__main__":
    main(overwrite=False)
