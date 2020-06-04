from flask import render_template, redirect, url_for, abort, flash, request,\
    current_app, make_response, send_from_directory
from werkzeug.utils import secure_filename

from . import main
from .. import cache
from ..models import save_access
import logging
from .forms import UploadForm
import os


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


@main.route('/')
def index():
    apps = [
        {'name': 'COIVD19 screening from X-rays', 'link': '/covid_form/'},

    ]
    return render_template('index.html', apps=apps)


@main.route('/covid_form/', methods=['GET', 'POST'])
def app_in_iframe():
    form = UploadForm()

    save_access(page='covid_form', access_type='homepage')

    # if form data is valid, go to success
    if form.validate_on_submit():
        file = form.data_file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        flash(f'Thank you {form.name.data}, Data was uploaded successfully')

        # return redirect('/thank_you')

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
