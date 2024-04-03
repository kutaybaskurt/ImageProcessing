from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import uuid

app = Flask(__name__)

# Model ve klasör yollarını belirleyin
model_path = '/Users/kutaybaskurt/Desktop/Engineering Project/Models And Graphics/xception.h5'
UPLOAD_FOLDER = '/Users/kutaybaskurt/Desktop/Engineering Project/Website/uploads'
RESULTS_FOLDER = '/Users/kutaybaskurt/Desktop/Engineering Project/Website/static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Modeli yükleyin
model = load_model(model_path)

# Dosya uzantısının izin verilen uzantılardan biri olup olmadığını kontrol eden yardımcı fonksiyon
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            # Benzersiz bir dosya adı oluştur
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = str(uuid.uuid4()) + '.' + ext
            
            # Dosyayı kaydet
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Resmi sınıflandır
            predicted_class, confidence = classify_image(file_path)
            
            # Yüklenen resmi sonuçlar klasörüne taşı
            result_path = os.path.join(app.config['RESULTS_FOLDER'], unique_filename)
            os.rename(file_path, result_path)
            
            # Sonuç sayfasını render et ve gerekli değişkenleri geç
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence, image_filename=unique_filename)

    # GET isteği için index sayfasını render et
    return render_template('index.html')

# Resmi ön işleme ve sınıflandırma için gerekli fonksiyonlar
def classify_image(image_path):
    # Resmi yükle ve ön işleme yap
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model ile tahmin yap
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    
    # Sınıflandırma sonucunu al
    class_labels = ['maize', 'rice', 'sugarcane', 'sunflower', 'wheat']
    predicted_class = class_labels[class_index]
    
    return predicted_class, confidence

if __name__ == '__main__':
    app.run(debug=True)
