import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from covid_xrays_model.config import config
from covid_xrays_model import __version__ as _version
import logging
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.vision import load_learner, Learner, ImageDataBunch, get_transforms, imagenet_stats
import fastai
import torch
import pathlib


logger = logging.getLogger(__name__)


learner_cache = {}


def load_dataset(*, sample_size=600, image_size=224, percent_gan_images=None) -> ImageDataBunch:

    labels = pd.read_csv(config.PROCESSED_DATA_DIR / 'labels_full.csv')

    classes = ['pneumonia', 'normal', 'COVID-19']
    ds_types = ['train']
    selected = []
    for c in classes:
        for t in ds_types:
            selected.extend(labels[(labels.label == c) & (labels.ds_type == t)][:sample_size].values.tolist())

    subset = pd.DataFrame(selected, columns=labels.columns)

    # Add GAN generated images to COVID dataset
    if percent_gan_images and percent_gan_images > 0:
        n_covid_images_needed = sample_size - subset[subset.label == "COVID-19"].shape[0]
        n_gan_sample = percent_gan_images * n_covid_images_needed // 100
        # walk through the directory and read GAN files names
        files = []
        for path in pathlib.Path(config.GAN_DATA_DIR).iterdir():
            sudirectory_path = '/'.join(path.parts[-2:])
            files.append(['', sudirectory_path, 'COVID-19', 'GAN', 'train'])

        gan_images = pd.DataFrame(files, columns=labels.columns).iloc[:n_gan_sample]
        subset = pd.concat([subset, gan_images])

    subset[['name', 'label']].to_csv(config.PROCESSED_DATA_DIR / 'labels.csv', index=False)

    tfms = _get_image_transformation()

    # will read from "labels.csv" in the data directory
    data = ImageDataBunch.from_csv(config.PROCESSED_DATA_DIR,
                                   ds_tfms=tfms,
                                   csv_labels=config.PROCESSED_DATA_DIR / 'labels.csv',
                                   valid_pct=0.1,
                                   seed=config.SEED,
                                   size=image_size,
                                   bs=21)

    data.normalize(imagenet_stats)

    return data


def _get_image_transformation():

    return get_transforms(do_flip=False,
                          max_lighting=0.1,
                          max_zoom=1.05,
                          max_warp=0.
                         )


def get_train_test_split(data: pd.DataFrame, test_size=config.TEST_SIZE, train_size=None):

    data = data.dropna(axis=0)

    # 100 equally spaced bins
    bins = np.linspace(0, data.shape[0], 100)
    # return the indices of the bins to which each value in target array belongs.
    y_binned = np.digitize(data[config.TARGET]/ 3600.0, bins)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.TARGET]/ 3600.0,
        test_size=test_size,
        train_size=train_size,
        stratify=y_binned,
        random_state=config.SEED)

    return X_train, X_test, y_train, y_test


def _get_postfix(with_focal_loss=False,
                 with_oversampling=False, sample_size=None, with_weighted_loss=False,
                 percent_gan_images=None):

    postfix = ''
    if with_oversampling:
        postfix += '_oversampling'
    if with_focal_loss:
        postfix += '_focus_loss'
    if with_weighted_loss:
        postfix += '_weighted_loss'
    if sample_size:
        postfix += f'_{sample_size}'
    if percent_gan_images:
        postfix+= f'_{percent_gan_images}'

    return postfix


def load_saved_learner(with_focal_loss=False, with_oversampling=False,
                       sample_size=None, with_weighted_loss=False, cpu=True, percent_gan_images=None):

    postfix = _get_postfix(with_focal_loss, with_oversampling, sample_size, with_weighted_loss, percent_gan_images)

    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}{postfix}.pkl'

    if cpu:  # map from gpu to
        fastai.torch_core.defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger.info(f"Loading model from: {save_file_name}")
    logger.info(f"Device: {fastai.torch_core.defaults.device}")

    if save_file_name in learner_cache:
        learn = learner_cache[save_file_name]
    else:
        learn = load_learner(config.TRAINED_MODEL_DIR, save_file_name)

    # layers_file = f'{config.TRAINED_MODEL_DIR}/{save_file_name}_layer_groups'
    # if fastai.torch_core.defaults.device == torch.device('cpu'):
    #     learn.layer_groups = torch.load(layers_file, map_location=torch.device('cpu'))
    # else:
    #     learn.layer_groups = torch.load(layers_file)

    # learn.layer_groups = joblib.load(filename=f'{config.TRAINED_MODEL_DIR}/{save_file_name}_layer_groups')

    return learn


def save_learner(learn: Learner, with_focal_loss=False,
                 with_oversampling=False, sample_size=None, with_weighted_loss=False,
                 percent_gan_images=''):

    postfix = _get_postfix(with_focal_loss, with_oversampling, sample_size, with_weighted_loss, percent_gan_images)

    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}{postfix}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    learn.export(save_path)

    # fix bug in fastai, missing layer_groups
    joblib.dump(learn.layer_groups, f'{save_path}_layer_groups')


def save_pipeline(*, pipeline_to_persist):
    """Persist the pipeline. """

    # Prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    logger.info(f'saved pipeline: {save_file_name}')


def current_model_exists():
    file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    path = config.TRAINED_MODEL_DIR / file_name

    return path.exists()

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]):
    """
    Remove old model pipelines. """

    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete and model_file.is_file():
            model_file.unlink()


def save_data(*, X, y, file_name : str, max_rows=-1):

    save_path = config.DATASET_DIR / file_name
    tmp = pd.concat([X, y], axis=1)

    if max_rows:
        tmp.iloc[:max_rows].to_csv(save_path, index=False)
    else:
        tmp.to_csv(save_path, index=False)

if __name__ == "__main__":
    # learn = load_saved_learner(with_focal_loss=False, with_oversampling=True, sample_size=5000)
    load_dataset(sample_size=5000, percent_gan_images=10)