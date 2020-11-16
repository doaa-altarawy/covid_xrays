import json
from base64 import b64encode
from app.models.users import User
from app.models.logs import Log
import pytest


def get_api_headers(username, password):
    return {
        'Authorization': 'Basic ' + b64encode(
            (username + ':' + password).encode('utf-8')).decode('utf-8'),
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

def test_404(flask_test_client):
    response = flask_test_client.get('/wrong/url')
    assert response.status_code == 404
    assert r'Not Found' in response.get_data(as_text=True)

def test_home(flask_test_client):

    response = flask_test_client.get('/')
    assert response.status_code == 200

def test_ml_datasets_app(flask_test_client):

    response = flask_test_client.get('/ml_datasets', follow_redirects=True)
    assert response.status_code == 200

def test_reaction_dataset_app(flask_test_client):

    response = flask_test_client.get('/covid_form', follow_redirects=True)
    assert response.status_code == 200



