import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

from app import create_app
import logging.config
import yaml

# IMPORTANT to set level to NONSET
logging.basicConfig(level=logging.NOTSET)
with open('logging_config.yml', 'r') as file:
    logging.config.dictConfig(yaml.full_load(file))

logger = logging.getLogger(__name__)

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

