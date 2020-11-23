import pandas as pd
import numpy as np
from covid_xrays_model.config import config
from covid_xrays_model.processing.data_management import load_saved_learner
import logging
import joblib
from fastai.vision import open_image, ClassificationInterpretation, tensor

logger = logging.getLogger(__name__)



def make_prediction_sample(image_path, cpu=True,
                           sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                           with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                           with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'],
                           with_weighted_loss=config.BEST_MODEL_PARAMS['with_weighted_loss'],):

    learn = load_saved_learner(sample_size=sample_size,
                               with_oversampling=with_oversampling,
                               with_focal_loss=with_focal_loss,
                               with_weighted_loss=with_weighted_loss,
                               cpu=cpu)

    # load image in grayscale
    image = open_image(image_path, convert_mode='L')
    cat = learn.predict(image)
    print(cat)

    class_prob = {k:float(v) for k, v in zip(learn.data.classes, cat[2])}

    return cat[0].obj, class_prob


def predict_dataset(ds_type: str='test', cpu=True,
                    sample_size=config.BEST_MODEL_PARAMS['sample_size'],
                    with_oversampling=config.BEST_MODEL_PARAMS['with_oversampling'],
                    with_focal_loss=config.BEST_MODEL_PARAMS['with_focal_loss'],
                    with_weighted_loss=config.BEST_MODEL_PARAMS['with_weighted_loss'],
                    confusion_matrix_filename='test_confusion_matrix',
                    percent_gan_images=None):

    learn = load_saved_learner(sample_size=sample_size,
                               with_oversampling=with_oversampling,
                               with_focal_loss=with_focal_loss,
                               with_weighted_loss=with_weighted_loss,
                               percent_gan_images=percent_gan_images,
                               cpu=cpu)


    classes = {c: i for i, c in enumerate(learn.data.classes)}

    # create a DF with filepath and class label (int)
    data = pd.read_csv(config.PROCESSED_DATA_DIR / 'labels_full.csv')
    data = data[data['ds_type'] == ds_type][['name', 'label']]
    data['label_int'] = data.label.apply(lambda x: classes[x])

    print(f'Running predictions on {data.shape[0]} data samples')

    data['pred_probability'] = pd.Series(dtype=object)
    for k, i in enumerate(data.index):
        pred = learn.predict(open_image(config.PROCESSED_DATA_DIR / data.loc[i, 'name'], convert_mode='L'))
        data.loc[i, 'y_pred'] = pred[0].data.item()
        data.at[i, 'pred_probability'] = pred[2].numpy()
        if k % 200 == 0 and k > 0:
            print(f'{k} images done..')

    data.to_csv(config.DATA_DIR / 'predictions.csv', index=False)

    print(f'Building classification interpretation..')
    interp = ClassificationInterpretation(learn=learn, losses=np.zeros(data.shape[0]),
                                          preds=tensor(data['pred_probability'].to_list()),
                                          y_true=tensor(data.label_int.to_list())
                                         )
    mat = interp.confusion_matrix()

    # sum diagonal / all data_size *100
    accuracy = np.trace(mat) / mat.sum() * 100

    print(mat)
    print(f'Accuracy: {accuracy}')

    interp.plot_confusion_matrix(return_fig=True).savefig(f'test_{confusion_matrix_filename}', dpi=200)
    joblib.dump(mat, f'test_{confusion_matrix_filename}.pkl')

    return mat, accuracy


if __name__ == "__main__":
    predict_dataset(ds_type='test')