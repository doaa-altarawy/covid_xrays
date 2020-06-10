import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import shutil
from covid_xrays_model.config import config
from covid_xrays_model import __version__ as _version
import logging
from typing import List
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.vision import load_learner, Learner, ImageDataBunch, get_transforms


logger = logging.getLogger(__name__)



def load_dataset(*, sample_size=300, image_size=200) -> ImageDataBunch:

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

    # will read from "labels.csv" in the data directory
    data = ImageDataBunch.from_csv(config.PROCESSED_DATA_DIR,
                                   ds_tfms=tfms,
                                   size=image_size)

    data.normalize()

    return data


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


def load_saved_learner():
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    learn = load_learner(config.TRAINED_MODEL_DIR, save_file_name)
    learn.layer_groups = joblib.load(filename=f'{config.TRAINED_MODEL_DIR}/{save_file_name}_layer_groups')

    return learn


def save_learner(learn: Learner):

    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
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
