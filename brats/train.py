import os
import sys
import glob

import tables

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)
sys.path.append(HOME)

from config import config
from unet3d.data import write_data_to_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model


def main(overwrite=False):
    hdf5_file_opened = tables.open_file(config["hdf5_file"], "r")

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"],
                              downsize_filters_factor=config["downsize_nb_filters_factor"],
                              pool_size=config["pool_size"], n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"])

    # get training and testing generators
    train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
        hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"], overwrite=overwrite,
        validation_keys_file=config["validation_file"], training_keys_file=config["training_file"],
        n_labels=config["n_labels"])

    # run training
    train_model(model=model, model_file=config["model_file"], training_generator=train_generator,
                validation_generator=validation_generator, steps_per_epoch=nb_train_samples,
                validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])
    hdf5_file_opened.close()


# if __name__ == "__main__":
#     main(overwrite=False)

if __name__ == '__main__':
    print(fetch_training_data_files())
