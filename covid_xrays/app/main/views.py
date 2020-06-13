from flask import render_template, redirect, url_for, abort, flash, request,\
    current_app, make_response, send_from_directory
from werkzeug.utils import secure_filename

from . import main
from .. import cache
from ..models import save_access
import logging
from .forms import UploadForm
import os
from pathlib import Path
from covid_xrays_model.predict import make_prediction_sample
from time import time

logger = logging.getLogger(__name__)

#  Logging to console in heroku
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)



@main.route('/shutdown')
def server_shutdown():
    if not current_app.testing:
        abort(404)
    shutdown = request.environ.get('werkzeug.server.shutdown')
    if not shutdown:
        abort(500)
    shutdown()
    return 'Shutting down...'


@main.route('/list')
def index():
    apps = [
        {'name': 'COIVD19 screening from X-rays', 'link': '/covid_form/'},

    ]
    return render_template('index.html', apps=apps)


@main.route('/', methods=['GET', 'POST'])
def covid_upload_form():
    form = UploadForm()

    save_access(page='covid_form', access_type='homepage')

    # if form data is valid, go to success
    if form.validate_on_submit():
        file = form.data_file.data

        # very important to avoid hacking, removes, ../../ from filenames
        filename = secure_filename(file.filename)
        # Add timestamp to filename
        filename = ".".join(filename.split('.')[:-1]) + str(time()) + '.' + filename.split('.')[-1]

        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

        # Create upload folder if doesn't exist
        if not Path(current_app.config['UPLOAD_FOLDER']).exists():
            Path(current_app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

        file.save(full_path)

        cat, prob = make_prediction_sample(full_path)

        logger.info(f'---------------------Cat: {cat}, {prob}')

        covid_prob = f'{prob["COVID-19"]:0.3f}%'
        suggested = ''
        if cat != 'COVID-19':
            suggested = f'is {cat}' if cat == 'normal' else f'has {cat} '
            suggested += f' (with {100*prob[cat]:0.2f}% confidence)'

        return render_template('covid/thank_you.html', covid_prob=covid_prob,
                               suggested=suggested, filename=filename)

    # return the empty form
    return render_template('covid/upload_data_form.html', form=form)



@main.route('/log_access/<access_type>/')
def log_download(access_type):

    logger.info('log_access: '.format(request.args))
    ds_name = request.args.get('dataset_name', None)
    ds_type = request.args.get('download_type', None)

    save_access(page='ml_datasets', access_type=access_type,
                dataset_name=ds_name, download_type=ds_type)

    return {'success': True}

@main.route('/uploads/<path:filename>')
def send_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)