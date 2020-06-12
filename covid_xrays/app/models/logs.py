import datetime
from .. import db
from flask import request, current_app
from .geo_location_util import get_geoip2_data


class Log(db.DynamicDocument):   # flexible schema, can have extra attributes
    """
    Stores searches and downloads of pages and apps
    Each Log is an access to specific page or resource


    Attributes
    ----------
    page: str
        The page the user accessed (like ml_download, app1, ..etc)

    access_type:
        'homepage', 'download', ...

    ip_address: str
        IP address of the cclient requesting the basis set
    date: datetime
        date of the search/download in the server local time

    """

    # access info
    page = db.StringField()
    access_type = db.StringField()

    dataset_name = db.StringField()
    download_type = db.StringField()

    date = db.DateTimeField(default=datetime.datetime.utcnow)

    # user info
    user_agent = db.StringField(max_length=512)
    header_email = db.StringField(max_length=100)
    ip_address = db.StringField()
    referrer = db.StringField(max_length=512)


    # extra computed geo data
    city = db.StringField()
    country = db.StringField()
    country_code = db.StringField()
    ip_lat = db.StringField()
    ip_long = db.StringField()
    postal_code = db.StringField()
    subdivision = db.StringField()

    meta = {
        'strict': False,     # allow extra fields
        'indexes': [
            "page", "access_type", "date"
        ]
    }

    def __str__(self):
        return 'Page: ' + str(self.page) \
               + ', access_type: ' + str(self.access_type) \
               + ', ip_address: ' + str(self.ip_address) \
               + ', referrer: ' + str(self.referrer) \
               + ', date: ' + str(self.date)


def save_access(page, access_type, **kwargs):

    if not current_app.config['DB_LOGGING']:
        return

    # The IP address is the last address listed in access_route, which
    # comes from the X-FORWARDED-FOR header
    # (If access_route is empty, use the original request ip)
    if len(request.access_route) > 0:
        ip_address = request.access_route[-1]
    else:
        ip_address = request.remote_addr

    user_agent = request.environ.get('HTTP_USER_AGENT', None)
    header_email = request.environ.get('HTTP_FROM', None)
    referrer = request.referrer

    # trim referrer and user agent if necessary
    if referrer is not None and len(referrer) > 512:
        referrer = referrer[:512]
    if user_agent is not None and len(user_agent) > 512:
        user_agent = user_agent[:512]

    # extra geo data
    extra = get_geoip2_data(ip_address)

    log = Log(page=page,
              access_type=access_type,
              ip_address=ip_address,
              user_agent=user_agent,
              header_email=header_email,
              referrer=referrer,
              **extra,
              **kwargs)

    log.save()
