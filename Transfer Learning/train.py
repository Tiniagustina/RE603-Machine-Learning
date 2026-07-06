import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# dataset 
base_dir = 'TrashType_Image_Dataset' 

print("Memulai persiapan data...")

# HANDLING OVERFITTING 1
train_datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    horizontal_flip=True,     
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

# Transfer Learning MobileNetV2
print("Membangun model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
# Dropout Layer
x = Dropout(0.5)(x) # Mematikan 50% neuron secara acak agar tidak menghafal

predictions = Dense(6, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
# Berhenti otomatis jika model mulai overfitting (val_loss naik)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5. Melatih Model
print("Mulai proses training...")
history = model.fit(
    train_generator,
    epochs=15, # Kita naikkan jadi 15, early_stop akan memberhentikan jika perlu
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Simpan Model
model.save('model_deteksi_sampah_anti_overfit.h5')
print("Model berhasil disimpan!")


# cek overfitting
plt.figure(figsize=(12, 4))

# Grafik Akurasi (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

# Grafik Kerugian (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Grafik Loss Model (Cek Overfitting)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Simpan grafik sebagai sebagai gambar
plt.savefig('grafik_evaluasi.png')
print("Grafik evaluasi berhasil disimpan sebagai 'grafik_evaluasi.png'.")
plt.show()