from covid_xrays_model.config import config
import numpy as np
np.random.seed(config.SEED)

from covid_xrays_model.processing.data_management import save_learner, \
    load_saved_learner, load_dataset
import logging
from torch import nn, tensor
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from fastai.vision import models, cnn_learner, torch, accuracy, ClassificationInterpretation, \
                          Learner, partial
from fastai.callbacks import OverSamplingCallback


logger = logging.getLogger(__name__)


def run_training_sample(sample_size=300, image_size=420, n_cycles=10,
                        with_focal_loss=True, with_oversampling=True,
                        with_weighted_loss=False):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    callbacks = None
    if with_oversampling:
        callbacks = [partial(OverSamplingCallback)]

    learn = cnn_learner(data, models.resnet50, metrics=accuracy,
                        callback_fns = callbacks
                        )
    learn.model = torch.nn.DataParallel(learn.model)

    # handle unbalanced data with weight
    # ['COVID-19', 'normal', 'pneumonia']
    if with_focal_loss:
        learn.loss_func = FocalLoss()
    elif with_weighted_loss:
        classes = {c:1 for c in learn.data.classes}
        classes['COVID-19'] = 2
        learn.loss_func = CrossEntropyLoss(weight=tensor(list(classes.values()), dtype=torch.float),
                                           reduction='mean')

    learn.fit_one_cycle(n_cycles)

    # learn.save('stage-50')
    # learn.load('stage-50')

    save_learner(learn, with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                 sample_size=sample_size, with_weighted_loss=with_weighted_loss)

    _save_classification_interpert(learn)


def plot_learning_rate(sample_size=300, image_size=420, load_learner=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    if load_learner:
        learn = load_saved_learner()
        learn.data = data
    else:
        learn = cnn_learner(data, models.resnet50, metrics=accuracy)
        learn.model = torch.nn.DataParallel(learn.model)

    learn.lr_find()
    learn.recorder.plot(return_fig=True).savefig('learning_rate.png', dpi=200)


def improve_saved_model(sample_size=300, image_size=420, n_cycles=5,
                        max_lr=slice(1e-6,1e-4), save=True,
                        with_focal_loss=True, with_oversampling=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    learn = load_saved_learner(with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                               sample_size=sample_size)
    learn.data = data

    learn.unfreeze()
    learn.fit_one_cycle(n_cycles, max_lr=max_lr)

    if save:
        save_learner(learn, with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                 sample_size=sample_size)

    _save_classification_interpert(learn)


def _save_classification_interpert(learn: Learner):

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(return_fig=True).savefig('confusion_matrix.png', dpi=200)
    interp.plot_top_losses(9, return_fig=True, figsize=(15,15)).savefig('top_losses.png', dpi=200)



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


if __name__ == '__main__':
   run_training_sample(sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                       image_size=config.BEST_MODEL_PARAMS['image_size'],
                       n_cycles=config.BEST_MODEL_PARAMS['n_cycles'],
                       with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                       with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'])

   # plot_learning_rate(sample_size=600, image_size=420)
   # improve_saved_model(sample_size=600, image_size=420, n_cycles=5, max_lr=slice(1e-6,1e-4), save=False)

