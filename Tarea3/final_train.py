import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import pickle

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

def build_model(dropout):
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    if dropout == 1:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(5, activation='softmax'))
    return model

#   MAIN

BASE_DIR = './img_divided'
SAVE_DIR = './final_models/'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

train_ds, val_ds = get_datasets(128)

#   Entrenar modelo sin dropout
print("--- Entrenando Modelo SIN Dropout ---")
model = build_model(0)
model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=1
    )

#   Guardar Modelo
model.save(os.path.join(SAVE_DIR, 'modelo_no_dropout.keras'))
#   Guardar Historial (Diccionario con los datos para graficar luego)
with open(os.path.join(SAVE_DIR, 'history_no_dropout.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

#   Entrenar Modelo con Dropout
print("\n--- Entrenando Modelo CON Dropout ---")
model_dropout = build_model(1)
model_dropout.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_dropout = model_dropout.fit(train_ds, validation_data=val_ds, epochs=20, verbose=1)

# Guardar Modelo
model_dropout.save(os.path.join(SAVE_DIR, 'modelo_dropout.keras'))
# Guardar Historial
with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'wb') as f:
    pickle.dump(history_dropout.history, f)

print(f"\n¡Entrenamiento finalizado! Modelos e historiales guardados en {SAVE_DIR}")