from flask import Flask
from flask_bootstrap import Bootstrap
from flask_mail import Mail
from flask_login import LoginManager
from flask_moment import Moment
from flask_pagedown import PageDown
from config import config
from flask_caching import Cache
from flask_cors import CORS
from flask_mongoengine import MongoEngine
from flask_admin import Admin
import logging

logger = logging.getLogger(__name__)

db = MongoEngine()
app_admin = Admin(name='COVID X-rays Admin', template_mode='bootstrap3',
                  base_template='admin/custom_base.html')


bootstrap = Bootstrap()
mail = Mail()
pagedown = PageDown()
moment = Moment()  # formatting dates and time
cache = Cache()
cors = CORS()

login_manager = LoginManager()
login_manager.login_view = 'auth.login'

def create_app(config_name):
    logger.info(f"logger: Creating flask app with config {config_name}")

    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    logger.info(f'Using Cache Type: {app.config["CACHE_TYPE"]}')

    bootstrap.init_app(app)
    mail.init_app(app)
    moment.init_app(app)
    if app.config['DB_LOGGING']:
        db.init_app(app)
    login_manager.init_app(app)
    pagedown.init_app(app)
    cache.init_app(app)
    cors.init_app(app)


    if app.config['SSL_REDIRECT']:
        from flask_sslify import SSLify
        sslify = SSLify(app)

    with app.app_context():

        logger.debug("Adding blueprints..")
        # The main application entry
        from .main import main as main_blueprint
        app.register_blueprint(main_blueprint)

        # For authentication
        from .auth import auth as auth_blueprint
        app.register_blueprint(auth_blueprint, url_prefix='/auth')

              # create user roles
        if app.config['DB_LOGGING']:
            from .models.users import update_roles
            update_roles()

        # To avoid circular import
        from app.admin import add_admin_views
        add_admin_views()

        # Then init the app
        app_admin.init_app(app)


        # Compile assets (JS, SCSS, less)
        logger.debug('Creating assets..')
        from .assets import compile_assets
        compile_assets(app)

        return app

    return app

