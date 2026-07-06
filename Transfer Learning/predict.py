import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Muat (Load) model yang sudah kita latih tadi
print("Sedang memuat model...")
model = tf.keras.models.load_model('model_deteksi_sampah.h5')

# 2. Daftar kelas sampah (harus urut abjad sesuai nama folder di datasetmu)
# Asumsinya ada 6 kelas: cardboard, glass, metal, paper, plastic, trash
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 3. Path ke foto yang ingin ditebak
# UBAH NAMA FILE INI sesuai dengan foto yang kamu siapkan!
img_path = 'test_kardus.jpg' 

def predict_trash(img_path):
    print(f"Menganalisis gambar: {img_path}...")
    
    # Muat gambar dan ubah ukurannya menjadi 224x224 (sesuai saat training)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Ubah gambar menjadi array angka dan normalisasi (dibagi 255)
    img_array = image.img_to_array(img) / 255.0
    
    # Tambahkan dimensi ekstra (karena model meminta format batch)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Lakukan prediksi
    predictions = model.predict(img_array)
    
    # Cari nilai prediksi tertinggi
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(predictions)
    
    print("\n" + "="*40)
    print(f" HASIL PREDIKSI:")
    print(f" Jenis Sampah : {class_names[class_idx].upper()}")
    print(f" Tingkat Keyakinan : {100 * np.max(score):.2f}%")
    print("="*40 + "\n")

# Jalankan fungsinya
predict_trash(img_path)