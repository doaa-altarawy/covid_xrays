import pathlib
import covid_xrays_model
import os


SEED = 0

PACKAGE_ROOT = pathlib.Path(covid_xrays_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

# Best model deployed on AWS
BEST_MODEL_PARAMS = {
    'sample_size': 5000,
    'image_size': 420,  # 224
    'n_cycles': 10,
    'with_focal_loss': False,
    'with_oversampling': True,
    'with_weighted_loss': True,
}


TESTING_DATA_FILE = 'test_data.csv'  # for pytest only

PIPELINE_NAME = 'covid_xrays_model'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

DATA_DIR = pathlib.Path(os.environ.get('DATA_DIR') or PACKAGE_ROOT.parents[1] / 'data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
GAN_DATA_DIR = PROCESSED_DATA_DIR / 'GAN_images'

# -------------------------  datasets --------------------------

TRAIN_FILE = PROCESSED_DATA_DIR / 'train_split_v3.txt'
TEST_FILE = PROCESSED_DATA_DIR / 'test_split_v3.txt'

# Raw data path within the data dir
# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = RAW_DATA_DIR / 'covid-chestxray-dataset/images'
cohen_csvpath = RAW_DATA_DIR / 'covid-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = RAW_DATA_DIR / 'Figure1-COVID-chestxray-dataset/images'
fig1_csvpath = RAW_DATA_DIR / 'Figure1-COVID-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://github.com/agchung/Actualmed-COVID-chestxray-dataset
actmed_imgpath = RAW_DATA_DIR / 'Actualmed-COVID-chestxray-dataset/images'
actmed_csvpath = RAW_DATA_DIR / 'Actualmed-COVID-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
sirm_imgpath = RAW_DATA_DIR / 'COVID-19-Radiography-Database/COVID-19'
sirm_csvpath = RAW_DATA_DIR / 'COVID-19-Radiography-Database/COVID-19.metadata.xlsx'

# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
rsna_datapath = RAW_DATA_DIR / 'rsna-pneumonia-detection-challenge'
# get all the normal from here
rsna_csvname = RAW_DATA_DIR / rsna_datapath / 'stage_2_detailed_class_info.csv'
# get all the 1s from here since 1 indicate pneumonia
# found that images that aren't pneunmonia and also not normal are classified as 0s
rsna_csvname2 = RAW_DATA_DIR / rsna_datapath / 'stage_2_train_labels.csv'
rsna_imgpath = RAW_DATA_DIR / rsna_datapath / 'stage_2_train_images'


# For differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 1e-2

TEST_SIZE = 0.2


