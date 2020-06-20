import pytest

from covid_xrays_model.predict import make_prediction_sample
from covid_xrays_model.processing.data_management import load_dataset
from covid_xrays_model import config
from covid_xrays_model import __version__ as _version


def test_single_prediction_pass(trained_model):

    # easy sample images
    sample_normal = './images/sample_normal.png'
    sample_covid = './images/sample_covid.jpg'

    pred, prob = make_prediction_sample(sample_normal)
    print(pred, prob)

    assert pred == 'normal'

    pred, prob = make_prediction_sample(sample_covid)
    print(pred, prob)

    assert pred == 'COVID-19'


#



