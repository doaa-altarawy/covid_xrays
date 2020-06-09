import pandas as pd
import numpy as np
from covid_xrays_model.config import config
from covid_xrays_model.processing.data_management import load_saved_learner
from covid_xrays_model.metrics import mape, percentile_rel_90
import logging
from typing import Union, List
from fastai.vision import open_image

logger = logging.getLogger(__name__)



def make_prediction_sample(image_path):

    learn = load_saved_learner()

    image = open_image(image_path)
    cat = learn.predict(image)
    print(cat)

    class_prob = {k:float(v) for k, v in zip(learn.data.classes, cat[2])}

    return cat[0].obj, class_prob
