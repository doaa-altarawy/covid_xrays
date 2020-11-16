import json
from flask import current_app
from flask_login import current_user
import pytest
from base64 import b64encode
from app.models.users import User, Permission


auth_url = '/auth'

# prerequisite for all other tests
@pytest.fixture(scope='module')
def add_admin(app):
    """Fill the test DB with the admin user
        The admin is user whose email is defined in Flask
        config in APP_ADMIN
        - must take app param to have application context
        - autouse to run by default without usage and make available
        - Can't use setup_class, app will be missing
    """
    User.objects.delete()
    user = User(email='daltarawy@vt.edu',
                username='Doaa Test', confirmed=True)
    user.password = 'fakePass'
    user.save()

    yield user

    User.objects().delete()


@pytest.fixture(scope='function')
def temp_user(app):
    user = User(email='test_user@vt.edu',
                username='Test', confirmed=True)
    user.password = 'fakePass'
    user.save()

    yield user

    User.objects(email=user.email).delete()


@pytest.fixture(scope='function')
def login_admin(add_admin, flask_test_client):
    flask_test_client.get(auth_url+'/logout', follow_redirects=True)
    data = dict(email='daltarawy@vt.edu', password='fakePass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                           follow_redirects=True)
    success = 'Change Email' in response.get_data(as_text=True)

    yield success

    # logout
    flask_test_client.get(auth_url+'/logout', follow_redirects=True)

# ---------------------------------------------------------

def test_app_is_testing(flask_test_client):
    assert current_app.config['TESTING']

def test_admin_user(login_admin, flask_test_client):
    user = User.objects(email='daltarawy@vt.edu').first()
    assert user
    assert user.is_administrator()
    assert user.can(Permission.MODERATE)

    with pytest.raises(AttributeError):
        user.password()

    assert user.to_json()['email'] == 'daltarawy@vt.edu'

    # admin is confirmed
    response = flask_test_client.get(auth_url+'/unconfirmed', follow_redirects=True)
    assert response.status_code == 200
    assert 'You have not confirmed your account yet' \
           not in response.get_data(as_text=True)

def test_app_exists():
    assert current_app is not None

def test_database_filled():
    assert User.objects(email='daltarawy@vt.edu').count() == 1

def get_api_headers(username, password):
    return {
        'Authorization': 'Basic ' + b64encode(
            (username + ':' + password).encode('utf-8')).decode('utf-8'),
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

def test_anonymous_user(flask_test_client):
    """Test the deafult anonymous user without login"""

    flask_test_client.get(auth_url+'/logout', follow_redirects=True)
    response = flask_test_client.get('/admin', follow_redirects=True)
    assert response.status_code == 200
    assert 'COVID X-rays Admin' in response.get_data(as_text=True)
    assert not current_user.is_administrator()
    assert not current_user.can(Permission.READ)


def test_register(flask_test_client):
    # get registration form
    response = flask_test_client.get((auth_url+'/register'))
    assert response.status_code == 200

    data = dict(email='dina@gmail.com', password='somePass',
                password2='somePass', username='Dina')
    response = flask_test_client.post(auth_url+'/register', data=data)
    # on success, redirect to home
    assert response.status_code == 302
    assert '/auth/login' in response.get_data(as_text=True)

    # test unconfirmed
    flask_test_client.get(auth_url+'/logout', follow_redirects=True)

    data = dict(email='dina@gmail.com', password='somePass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'You have not confirmed your account yet' \
           in response.get_data(as_text=True)

    # confirm email
    response = flask_test_client.get(auth_url+'/confirm', follow_redirects=True)
    assert 'You have not confirmed your account yet' \
           in response.get_data(as_text=True)

    # wrong confirmation token
    response = flask_test_client.get(auth_url+'/confirm/'+'wrong',
                          follow_redirects=True)
    assert 'The confirmation link is invalid or has expired' \
           in response.get_data(as_text=True)

    user = User.objects(email='dina@gmail.com').first()
    token = user.generate_confirmation_token()
    response = flask_test_client.get(auth_url+'/confirm/'+token,
                          follow_redirects=True)
    assert 'You have confirmed your account. Thanks' \
           in response.get_data(as_text=True)
    # update user from DB, and check if confirmed
    assert User.objects(email='dina@gmail.com').first().confirmed

def test_register_exiting_email(flask_test_client):
    data = dict(email='daltarawy@vt.edu', password='somePass',
                password2='somePass', username='Doaa')
    response = flask_test_client.post(auth_url+'/register', data=data)
    assert response.status_code == 200
    assert 'Email already registered' in response.get_data(as_text=True)

    data = dict(email='daltarawy@vt.edu', password='fakePass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                                follow_redirects=True)
    # on success, redirect to home
    assert response.status_code == 200
    assert '/admin/' in response.get_data(as_text=True)
    assert 'Change Email' in response.get_data(as_text=True)
    assert 'Change Password' in response.get_data(as_text=True)

    # Try to confirmation already confirmed, redirect to home
    response = flask_test_client.get(auth_url+'/confirm/'+'123',
                          follow_redirects=True)
    assert '/admin/' in response.get_data(as_text=True)

def test_wrong_sign_in(flask_test_client):
    data = dict(email='daltarawy@vt.edu', password='wrongPass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'Invalid email or password' in response.get_data(as_text=True)

def test_sign_out(flask_test_client):
    # make sure you are logged in first
    data = dict(email='daltarawy@vt.edu', password='fakePass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'Change Email' in response.get_data(as_text=True)
    # logout
    response = flask_test_client.get(auth_url+'/logout',
                          follow_redirects=True)
    assert 'You have been logged out' in response.get_data(as_text=True)

def test_change_password(login_admin, flask_test_client):
    # log in as admin
    response = flask_test_client.get(auth_url+'/change-password')
    assert response.status_code == 200
    assert 'Old password' in response.get_data(as_text=True)

    # new password does not match
    data = dict(old_password='wrong', password='123', password2='456')
    response = flask_test_client.post(auth_url+'/change-password', data=data)
    assert response.status_code == 200
    assert 'Passwords must match' in response.get_data(as_text=True)

    # wrong old password
    data = dict(old_password='wrong', password='1234567', password2='1234567')
    response = flask_test_client.post(auth_url+'/change-password', data=data)
    assert response.status_code == 200
    assert 'Invalid password' in response.get_data(as_text=True)

    # correct entry
    data = dict(old_password='fakePass', password='1234567', password2='1234567')
    response = flask_test_client.post(auth_url+'/change-password', data=data,
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'Your password has been updated' in response.get_data(as_text=True)

    # change password back to original, redirect of success
    data = dict(old_password='1234567', password='fakePass', password2='fakePass')
    response = flask_test_client.post(auth_url+'/change-password', data=data)
    assert response.status_code == 302

# @pytest.mark.skip
def test_change_email(temp_user, flask_test_client):
    # log in as temp_user
    data = dict(email=temp_user.email, password='fakePass')
    response = flask_test_client.post(auth_url+'/login', data=data,
                           follow_redirects=True)
    assert 'Change Email' in response.get_data(as_text=True)

    # Email already exist
    data = dict(email=temp_user.email, password='fakePass')
    response = flask_test_client.post(auth_url+'/change_email', data=data)
    assert response.status_code == 200
    assert 'Email already registered' in response.get_data(as_text=True)

    # Wrong password
    data = dict(email='temp2@vt.edu', password='wrong')
    response = flask_test_client.post(auth_url+'/change_email', data=data)
    assert response.status_code == 200
    assert 'Invalid password' in response.get_data(as_text=True)

    # Wrong Confirm changed email
    token = 'Wrong token'
    url = auth_url + '/change_email/' + token
    response = flask_test_client.get(url, follow_redirects=True)
    assert 'Your email address has been updated' not in response.get_data(as_text=True)

    # correct entry to change email
    data = dict(email='temp2@vt.edu', password='fakePass')
    response = flask_test_client.post(auth_url+'/change_email', data=data,
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'confirm your new email address' in response.get_data(as_text=True)

    # Confirm changed email
    user = User.objects(email=temp_user.email).first()
    token = user.generate_email_change_token(new_email='temp2@vt.edu')
    url = auth_url + '/change_email/' + token
    response = flask_test_client.get(url, follow_redirects=True)
    assert 'Your email address has been updated' in response.get_data(as_text=True)

def test_reset_password(login_admin, flask_test_client):
    # if already logged in, redirect to home
    response = flask_test_client.get(auth_url+'/reset', follow_redirects=True)
    assert 'Change Email' in response.get_data(as_text=True)

    # Logout and reset password
    flask_test_client.get(auth_url+'/logout', follow_redirects=True)

    response = flask_test_client.get(auth_url+'/reset', follow_redirects=True)
    assert response.status_code == 200
    assert 'Reset Your Password' in response.get_data(as_text=True)

    response = flask_test_client.post(auth_url+'/reset',
                           data=dict(email='someEmail@vt.edu'),
                           follow_redirects=True)
    assert response.status_code == 200
    assert 'An email with instructions to reset your password' \
           in response.get_data(as_text=True)

    # clicking the link in the email and changing the password
    user = User.objects(email='daltarawy@vt.edu').first()
    token = user.generate_reset_token()
    url = auth_url + '/reset/' + token
    response = flask_test_client.post(url,
                           data=dict(password='fakePass', password2='fakePass'),
                           follow_redirects=True)
    assert 'Your password has been updated' in response.get_data(as_text=True)

def test_auth_token(login_admin):
    user = User.objects(email='daltarawy@vt.edu').first()
    assert str(user) == 'username=daltarawy@vt.edu'
    assert repr(user) == "<User 'daltarawy@vt.edu'>"
    token = user.generate_auth_token()
    assert User.verify_auth_token(token).id == user.id
    assert not User.verify_auth_token('wrong token')
