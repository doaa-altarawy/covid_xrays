from flask_admin.contrib.mongoengine import ModelView
from ..models.logs import Log
import flask_login as login
from datetime import date, datetime
from flask_admin.model import typefmt
from ..models.users import User, Permission, Role


def date_format(view, value):
    return value.strftime('%Y-%m-%d %H:%M')


MY_DEFAULT_FORMATTERS = dict(typefmt.BASE_FORMATTERS)
MY_DEFAULT_FORMATTERS.update({
        date: date_format
    })


class UserView(ModelView):

    can_create = False
    column_type_formatters = MY_DEFAULT_FORMATTERS
    column_list = ['username', 'email', 'role']
    form_excluded_columns = ['password_hash', 'avatar_hash', 'location', 'confirmed']

    inline_models = (Role, )

    form_widget_args = dict(
        email={'readonly': True},
        username={'readonly': True},
        member_since={'readonly': True}
    )

    def is_accessible(self):
        return login.current_user.is_authenticated and login.current_user.can(Permission.ADMIN)


class LogView(ModelView):

    can_create = False
    can_edit = True
    can_export = True
    can_view_details = True

    column_type_formatters = MY_DEFAULT_FORMATTERS
    column_exclude_list = []
    column_filters = ['page', 'access_type', 'dataset_name']

    # Bug in Flask admin, don't use disabled: True
    # Bug in Flask admin, readonly doesn't work with boolean and ListFields
    form_widget_args = dict(
        date={'readonly': True},
        page={'readonly': True},
        access_type={'readonly': True},
        dataset_name={'readonly': True},
        user_agent={'readonly': True},
        header_email={'readonly': True},
        city={'readonly': True},
        country={'readonly': True},
        subdivision={'readonly': True},
        ip_address={'readonly': True}
    )

    def is_accessible(self):
        return login.current_user.is_authenticated and login.current_user.can(Permission.ADMIN)


def add_admin_views():
    """Register views to admin"""
    from .. import app_admin
    app_admin.add_view(LogView(Log, name='Access Log'))
    app_admin.add_view(UserView(User, name='Users'))
