import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    # Flask-Caching configs
    # In dev, default to use memory cache
    CACHE_TYPE = os.environ.get('CACHE_TYPE') or 'simple'
    CACHE_DEFAULT_TIMEOUT = 60 * 60 * 24  # in seconds, one day
    WTF_CSRF_TIME_LIMIT = 60 * 60 *24

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'

    # Mail
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in \
                            ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[Covid19 screening app]'
    MAIL_SENDER = 'COVID-xrays Admin <doaa.altarawy@gmail.com>'

    SSL_REDIRECT = False

    # Admin login
    APP_ADMIN = os.environ.get('APP_ADMIN', 'daltarawy@gmail.com')
    EMAIL_CONFIRMATION_ENABLED = False

    # log page access to db or not
    DB_LOGGING = False

    UPLOAD_FOLDER = basedir + '/uploads'

    MONGODB_SETTINGS = {
        'host': os.environ.get('MONGO_URI',
                               "mongodb://<dbuser>:<dbpassword>localhost::27017/covid_xrays_logging_db"),
    }

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    ASSETS_DEBUG = True


class TestingConfig(Config):
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    EMAIL_CONFIRMATION_ENABLED = True
    # disable CSRF protection in testing
    WTF_CSRF_ENABLED = False
    MONGODB_SETTINGS = {
        'db': "test_covid_db",
    }


class ProductionConfig(Config):

    CACHE_TYPE = os.environ.get('CACHE_TYPE') or 'simple'


    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # email errors to the administrators
        import logging
        from logging.handlers import SMTPHandler
        credentials = None
        secure = None
        if getattr(cls, 'MAIL_USERNAME', None) is not None:
            credentials = (cls.MAIL_USERNAME, cls.MAIL_PASSWORD)
            if getattr(cls, 'MAIL_USE_TLS', None):
                secure = ()
        mail_handler = SMTPHandler(
            mailhost=(cls.MAIL_SERVER, cls.MAIL_PORT),
            fromaddr=cls.MAIL_SENDER,
            toaddrs=[cls.APP_ADMIN],
            subject=cls.MAIL_SUBJECT_PREFIX + ' Application Error',
            credentials=credentials,
            secure=secure)
        mail_handler.setLevel(logging.ERROR)
        # app.logger.addHandler(mail_handler)


class HerokuConfig(ProductionConfig):
    SSL_REDIRECT = True if os.environ.get('DYNO') else False

    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)

        # handle reverse proxy server headers
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app)

        # log to stderr
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        app.logger.addHandler(file_handler)


class DockerConfig(ProductionConfig):
    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)

        # log to stderr
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


class UnixConfig(ProductionConfig):
    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)

        # log to syslog
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.INFO)
        app.logger.addHandler(syslog_handler)


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'heroku': HerokuConfig,
    'docker': DockerConfig,
    'unix': UnixConfig,

    'default': DevelopmentConfig,
}
