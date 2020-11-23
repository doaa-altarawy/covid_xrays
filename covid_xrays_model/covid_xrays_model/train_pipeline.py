from covid_xrays_model.config import config
import numpy as np
np.random.seed(config.SEED)

from covid_xrays_model.processing.data_management import save_learner, \
    load_saved_learner, load_dataset
import logging
import joblib
from torch import nn, tensor
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from fastai.vision import models, cnn_learner, torch, accuracy, ClassificationInterpretation, \
                          Learner, partial
from fastai.callbacks import OverSamplingCallback
import fastai


logger = logging.getLogger(__name__)


def run_training_sample(sample_size=300, image_size=224, n_cycles=10,
                        with_focal_loss=False, with_oversampling=True,
                        with_weighted_loss=True,
                        confusion_matrix_filename='train_confusion_matrix',
                        percent_gan_images=0):
    """

    :param sample_size: number of images per class
            if the input file has less than the given number, only the existing ones are used
    :param image_size:
            Size of the image in image augmantation pre-processing
    :param n_cycles:
            epochs
    :param with_focal_loss: bool
        Use it if data is balanced
    :param with_oversampling: bool
        Use oversampling for the mintority class of COVID xrays to match the `sample_size`
    :param with_weighted_loss: bool
        Use weighted loss for unbalaned sample size in classes
    :param percent_gan_images: percent og GAN generated to use (between 0-100)
        If with_oversampling is True, the it's used after adding GAN images
    :return:
    """

    data = load_dataset(sample_size=sample_size, image_size=image_size,
                        percent_gan_images=percent_gan_images)

    callbacks = None
    if with_oversampling:
        callbacks = [partial(OverSamplingCallback)]

    learn = cnn_learner(data, models.resnet50, metrics=accuracy,
                        callback_fns = callbacks
                        )
    learn.model = torch.nn.DataParallel(learn.model)

    # handle unbalanced data with weights
    # ['COVID-19', 'normal', 'pneumonia']
    if with_weighted_loss:
        classes = {c:1 for c in learn.data.classes}
        classes['COVID-19'] = 5
        learn.loss_func = CrossEntropyLoss(weight=tensor(list(classes.values()),
                                           dtype=torch.float, device=fastai.torch_core.defaults.device),
                                           reduction='mean')
    elif with_focal_loss:
        learn.loss_func = FocalLoss()

    learn.fit_one_cycle(n_cycles)

    save_learner(learn, with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                 sample_size=sample_size, with_weighted_loss=with_weighted_loss,
                 percent_gan_images=percent_gan_images)

    _save_classification_interpert(learn, confusion_matrix_filename=confusion_matrix_filename)


def plot_learning_rate(sample_size=300, image_size=224, load_learner=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    if load_learner:
        learn = load_saved_learner()
        learn.data = data
    else:
        learn = cnn_learner(data, models.resnet50, metrics=accuracy)
        learn.model = torch.nn.DataParallel(learn.model)

    learn.lr_find()
    learn.recorder.plot(return_fig=True).savefig('learning_rate.png', dpi=200)


def improve_saved_model(sample_size=300, image_size=224, n_cycles=5,
                        max_lr=slice(1e-6,1e-4), save=False,
                        with_focal_loss=False, with_oversampling=True,
                        with_weighted_loss=True):

    data = load_dataset(sample_size=sample_size, image_size=image_size)

    learn = load_saved_learner(with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                               sample_size=sample_size, with_weighted_loss=with_weighted_loss)
    learn.data = data

    learn.unfreeze()
    learn.fit_one_cycle(n_cycles, max_lr=max_lr)

    if save:
        save_learner(learn, with_focal_loss=with_focal_loss, with_oversampling=with_oversampling,
                 sample_size=sample_size, with_weighted_loss=with_weighted_loss)

    _save_classification_interpert(learn)


def _save_classification_interpert(learn: Learner, confusion_matrix_filename='confusion_matrix'):

    # interp = ClassificationInterpretation.from_learner(learn)
    train_interp = learn.interpret(ds_type=fastai.vision.DatasetType.Train)
    valid_interp = learn.interpret(ds_type=fastai.vision.DatasetType.Valid)

    joblib.dump(train_interp.confusion_matrix(), f'train_{confusion_matrix_filename}.pkl')
    joblib.dump(valid_interp.confusion_matrix(), f'valid_{confusion_matrix_filename}.pkl')

    train_interp.plot_confusion_matrix(return_fig=True).savefig(f'train_{confusion_matrix_filename}', dpi=200)
    train_interp.plot_top_losses(9, return_fig=True, figsize=(14,14)).savefig('top_losses.png', dpi=200)


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
   run_training_sample(sample_size=100, #config.BEST_MODEL_PARAMS['sample_size'],
                       image_size=config.BEST_MODEL_PARAMS['image_size'],
                       n_cycles=config.BEST_MODEL_PARAMS['n_cycles'],
                       with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                       with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'])

   # plot_learning_rate(sample_size=600, image_size=420)
   # improve_saved_model(sample_size=600, image_size=420, n_cycles=5, max_lr=slice(1e-6,1e-4), save=False)

