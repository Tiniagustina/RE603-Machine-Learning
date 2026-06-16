import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# 1. Path ke folder dataset (Sesuaikan dengan struktur foldermu)
base_dir = os.path.join('TrashType_Image_Dataset')

print("Memulai persiapan data...")

# 2. Augmentasi & Preprocessing Data
train_datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    horizontal_flip=True,     
    validation_split=0.2      # Sisihkan 20% untuk validasi
)

# Memuat Data Training
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),   
    batch_size=32,
    class_mode='categorical', 
    subset='training'
)

# Memuat Data Validasi
val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 3. Membangun Model (MobileNetV2)
print("Membangun model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Kunci layer bawaan

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# Karena dataset ini memiliki 6 kelas (cardboard, glass, metal, paper, plastic, trash)
predictions = Dense(6, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Melatih Model
print("Mulai proses training...")
history = model.fit(
    train_generator,
    epochs=10,                      # Coba 10 putaran dulu
    validation_data=val_generator
)

# 5. Simpan Model setelah selesai
model.save('model_deteksi_sampah.h5')
print("Training selesai! Model berhasil disimpan sebagai 'model_deteksi_sampah.h5'")