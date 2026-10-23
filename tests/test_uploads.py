from io import BytesIO
import importlib
import numpy as np
from PIL import Image
import pytest

web = importlib.import_module('app.app')

def image_bytes():
    buffer = BytesIO()
    Image.new('L', (180, 180), 128).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def test_upload_contract(monkeypatch):
    client = web.app.test_client()
    assert client.get('/').status_code == 200
    assert client.post('/').status_code == 400
    assert client.post('/', data={'file1': (BytesIO(b'not an image'), 'bad.png')}).status_code == 400
    monkeypatch.setattr(web, 'predict', lambda _: 1)
    response = client.post('/', data={'file1': (image_bytes(), 'sample.png')})
    assert response.status_code == 200
    assert b'aria-live' in response.data

def test_model_failure_is_service_error(monkeypatch):
    def unavailable(_):
        raise RuntimeError('missing weights')
    monkeypatch.setattr(web, 'predict', unavailable)
    response = web.app.test_client().post('/', data={'file1': (image_bytes(), 'sample.png')})
    assert response.status_code == 503

def test_preprocessing_contract():
    from app.model import image_pre
    data = image_pre(image_bytes())
    assert data.shape == (1, 32, 32, 1)
    assert np.isfinite(data).all()
    assert data.min() >= 0 and data.max() <= 1

def test_real_shipped_baseline():
    from app.model import image_pre, predict
    assert predict(image_pre(image_bytes())) in (0, 1)
    response = web.app.test_client().post('/', data={'file1': (image_bytes(), 'sample.png')})
    assert response.status_code == 200
