import pandas as pd
import numpy as np
from covid_xrays_model.config import config
from covid_xrays_model.processing.data_management import load_saved_learner
import logging
import joblib
from fastai.vision import open_image, ClassificationInterpretation, tensor

logger = logging.getLogger(__name__)



def make_prediction_sample(image_path, cpu=True):

    learn = load_saved_learner(sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                               with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                               with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'],
                               with_weighted_loss=config.BEST_MODEL_PARAMS['with_weighted_loss'],
                               cpu=cpu)

    # load image in grayscale
    image = open_image(image_path, convert_mode='L')
    cat = learn.predict(image)
    print(cat)

    class_prob = {k:float(v) for k, v in zip(learn.data.classes, cat[2])}

    return cat[0].obj, class_prob


def predict_dataset(ds_type: str='test'):

    learn = load_saved_learner(sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                               with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                               with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'],
                               with_weighted_loss=config.BEST_MODEL_PARAMS['with_weighted_loss']
                               )

    classes = {c: i for i, c in enumerate(learn.data.classes)}

    # create a DF with filepath and class label (int)
    data = pd.read_csv(config.PROCESSED_DATA_DIR / 'labels_full.csv')
    data = data[data['ds_type'] == ds_type][['name', 'label']]
    data['label_int'] = data.label.apply(lambda x: classes[x])

    print(f'Running predictions on {data.shape[0]} data samples')

    data['y_pred'] = data.name.apply(lambda x:
                                     learn.predict(open_image(config.PROCESSED_DATA_DIR / x, convert_mode='L')))
    pred_tesnor = data.y_pred.apply(lambda x: x[2].tolist()).to_list()
    pred_tesnor = tensor(np.array(pred_tesnor))

    data.to_csv(config.PROCESSED_DATA_DIR / 'predictions.csv')

    print(f'Building classification interpretation..')
    interp = ClassificationInterpretation(learn=learn, losses=np.zeros(data.shape[0]),
                                                preds=pred_tesnor,
                                                y_true=tensor(data.label_int.to_list())
                                               )
    mat = interp.confusion_matrix()

    # sum diagonal / all data_size *100
    accuracy = np.trace(mat) / mat.sum() * 100

    print(mat)
    print(f'Accuracy: {accuracy}')

    interp.plot_confusion_matrix(return_fig=True).savefig(config.PROCESSED_DATA_DIR / 'confusion_matrix.png', dpi=200)
    joblib.dump(mat, config.PROCESSED_DATA_DIR / 'confusion_matrix.pkl')


if __name__ == "__main__":
    predict_dataset(ds_type='test')