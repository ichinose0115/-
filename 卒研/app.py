from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# アップロードされた画像の保存先
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# モデルの読み込み
model = joblib.load('new_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='ファイルがありません')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='ファイルが選択されていません')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).reshape(1, -1)

        prediction = model.predict(img_array)[0]

        result = "犬" if prediction == 0 else "猫"

        return render_template('index.html', message='この画像は {} です'.format(result), filename=filename)

    else:
        return render_template('index.html', message='許可されていない拡張子です')

if __name__ == '__main__':
    app.run(debug=True)
