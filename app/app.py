"""Validated, in-memory image upload demo."""
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
if __package__:
    from .model import image_pre, predict
else:
    from model import image_pre, predict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return render_template('index.html', error='The maximum upload size is 8 MB.'), 413

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    upload = request.files.get('file1')
    if upload is None or not upload.filename:
        return render_template('index.html', error='Choose a PNG or JPEG image.'), 400
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', error='Use a PNG or JPEG image.'), 400
    try:
        data = image_pre(upload.stream)
        prediction = predict(data)
    except (ValueError, OSError) as exc:
        return render_template('index.html', error=str(exc)), 400
    except RuntimeError as exc:
        app.logger.warning('Inference unavailable: %s', exc)
        return render_template('index.html', error='Prediction is unavailable. Check the model installation described in README.md.'), 503
    result = 'Model class: Healthy (training label 1).' if prediction == 1 else 'Model class: Brain_Tumor (training label 0).'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=False)
