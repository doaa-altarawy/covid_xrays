import pandas as pd
import numpy as np
from covid_xrays_model.config import config
from covid_xrays_model.processing.data_management import load_saved_learner
import logging
from fastai.vision import open_image

logger = logging.getLogger(__name__)



def make_prediction_sample(image_path, cpu=True):

    learn = load_saved_learner(sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                               with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                               with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'],
                               cpu=cpu)

    image = open_image(image_path)
    cat = learn.predict(image)
    print(cat)

    class_prob = {k:float(v) for k, v in zip(learn.data.classes, cat[2])}

    return cat[0].obj, class_prob
