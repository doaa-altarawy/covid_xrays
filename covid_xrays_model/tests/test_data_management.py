import pytest
from covid_xrays_model import config
from covid_xrays_model.processing.data_management import load_dataset
import pathlib
import os


def test_load_dataset():

    data = load_dataset(image_size=420, sample_size=5000)

    print(data)
