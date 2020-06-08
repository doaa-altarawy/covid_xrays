from covid_xrays_model.config import config
import numpy as np
np.random.seed(config.SEED)

from covid_xrays_model import __version__ as _version
import logging


from fastai.vision import *
from fastai.vision import models
import pandas as pd


logger = logging.getLogger(__name__)


def run_training_sample(sample_size=600, image_size=480):
    labels = pd.read_csv(config.PROCESSED_DATA_DIR / 'labels_full.csv')

    classes = ['pneumonia', 'normal', 'COVID-19']
    ds_types = ['train', 'test']
    selected = []
    for c in classes:
        for t in ds_types:
            selected.extend(labels[(labels.label == c) & (labels.ds_type == t)][:sample_size].values.tolist())

    subset = pd.DataFrame(selected, columns=labels.columns)
    subset[['name', 'label']].to_csv(config.PROCESSED_DATA_DIR / 'labels.csv', index=False)

    tfms = get_transforms(do_flip=False,
    #                       flip_vert=True,
    #                       max_lighting=0.1, max_zoom=1.05, max_warp=0.
                         )

    data = ImageDataBunch.from_csv(config.PROCESSED_DATA_DIR, ds_tfms=tfms, size=image_size)

    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.model = torch.nn.DataParallel(learn.model)

    learn.fit(10)

    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    learn.export(save_path)



if __name__ == '__main__':
    run_training_sample()
