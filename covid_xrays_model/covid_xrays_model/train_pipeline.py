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

    data = ImageDataBunch.from_csv(config.PROCESSED_DATA_DIR,
                                   ds_tfms=tfms,
                                   size=image_size)

    data.normalize()

    learn = cnn_learner(data, models.resnet50, metrics=accuracy)
    learn.model = torch.nn.DataParallel(learn.model)

    learn.lr_find()
    learn.recorder.plot(return_fig=True).savefig('learning_rate.png', dpi=200)

    learn.fit_one_cycle(10)
    learn.save('stage-50')

    # learn.load('stage-50')
    # learn.unfreeze()
    # learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    learn.export(save_path)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(return_fig=True).savefig('confusion_matrix.png', dpi=200)


if __name__ == '__main__':
    run_training_sample()
