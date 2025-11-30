import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from tqdm import tqdm

#   Obtención de dataset (tamaño específico)
def get_datasets(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'training_set'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        verbose=0
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'validation_set'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        verbose=0
    )
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

#   Construcción de modelo
def build_model(dropout_val):
    model = models.Sequential()

    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    #   Acrchitecture
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten()) #   Espacio latente
    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dropout(dropout_val))

    model.add(layers.Dense(5, activation='softmax')) # 5 Clases
    return model

#   MAIN

BASE_DIR = './img_divided'

#   Valores a probar
dropout_val = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
dropout_accuracy = []

print("\n" + "="*30)
print("Iniciando prueba de dropout")
print("="*30 + "\n")

train_ds, val_ds = get_datasets(128)
for i, val in enumerate(tqdm(dropout_val)):
    tqdm.write(f"Probando porcentaje de dropout = {val}")
    
    model = build_model(dropout_val=val)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=0
    )

    dropout_accuracy.append(max(history.history['val_accuracy']))
    tf.keras.backend.clear_session()
    del model

#   Imprimir resultados finales
print("\n" + "="*30)
print(" RESULTADOS FINALES ")
print("="*30 + "\n")

for i, val in enumerate(dropout_val):
    print(f"{i}. dropout porcentaje = {val}, Accuracy = {dropout_accuracy[i]:.4f}")