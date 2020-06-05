import pandas as pd
import numpy as np
from covid_xrays_model.config import config
from covid_xrays_model.processing.data_management import load_pipeline
from covid_xrays_model.metrics import mape, percentile_rel_90
from covid_xrays_model import __version__ as _version
import logging
from typing import Union, List
from fastai.vision import load_learner, open_image

logger = logging.getLogger(__name__)


def make_prediction_sample(image_path):

    learn = load_learner(config.TRAINED_MODEL_DIR, config.PIPELINE_SAVE_FILE + '.pkl')

    image = open_image(image_path)
    cat = learn.predict(image)
    print(cat)

    class_prob = {k:float(v) for k, v in zip(learn.data.classes, cat[2])}

    return cat[0].obj, class_prob
