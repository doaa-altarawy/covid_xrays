from covid_xrays_model.train_pipeline import run_training_sample
from covid_xrays_model.processing.data_management import load_saved_learner
from covid_xrays_model import config
from covid_xrays_model import __version__ as _version
import pathlib
import pytest
import logging


@pytest.fixture(scope='session')
def trained_model():

    params = config.BEST_MODEL_PARAMS

    # Simple model
    # params = {
    #         'sample_size': 600,
    #         'image_size': 300,
    #         'n_cycles': 10,
    #         'with_focal_loss': False,
    #         'with_oversampling': True
    # }

    try:
        load_saved_learner(with_focal_loss=params['with_focal_loss'],
                           with_oversampling=params['with_oversampling'],
                           sample_size=params['sample_size'],
                           cpu=True)
        return True
    except Exception as err:
        print(err)
        print("Running training to generate the model for tests")

        run_training_sample(sample_size=params['sample_size'],
                            image_size=params['image_size'],
                            n_cycles=params['n_cycles'],
                            with_oversampling=params['with_oversampling'],
                            with_focal_loss=params['with_focal_loss'])

    return True