from covid_xrays_model.config import config
import numpy as np
np.random.seed(config.SEED)

from covid_xrays_model.processing.data_management import save_learner, \
    load_saved_learner, load_dataset
import logging

from fastai.vision import models, cnn_learner, torch, accuracy, ClassificationInterpretation, \
                          Learner


logger = logging.getLogger(__name__)


def run_training_sample(sample_size=300, image_size=420, n_cycles=10):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    learn.model = torch.nn.DataParallel(learn.model)

    learn.fit_one_cycle(n_cycles)

    # learn.save('stage-50')
    # learn.load('stage-50')

    save_learner(learn)

    _save_classification_interpert(learn)


def plot_learning_rate(sample_size=300, image_size=420, load_learner=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    if load_learner:
        learn = load_saved_learner()
        learn.data = data
    else:
        learn = cnn_learner(data, models.resnet34, metrics=accuracy)
        learn.model = torch.nn.DataParallel(learn.model)

    learn.lr_find()
    learn.recorder.plot(return_fig=True).savefig('learning_rate.png', dpi=200)


def improve_saved_model(sample_size=300, image_size=420, n_cycles=5,
                     max_lr=slice(1e-6,1e-4), save=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    learn = load_saved_learner()
    learn.data = data

    learn.unfreeze()
    learn.fit_one_cycle(n_cycles, max_lr=max_lr)

    if save:
        save_learner(learn)

    _save_classification_interpert(learn)


def _save_classification_interpert(learn: Learner):

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(return_fig=True).savefig('confusion_matrix.png', dpi=200)
    interp.plot_top_losses(9, return_fig=True, figsize=(15,15)).savefig('top_losses.png', dpi=200)


if __name__ == '__main__':
   run_training_sample(sample_size=600, image_size=420, n_cycles=10)
   # plot_learning_rate(sample_size=600, image_size=420)
   # improve_saved_model(sample_size=600, image_size=420, n_cycles=5, max_lr=slice(1e-6,1e-4), save=False)

