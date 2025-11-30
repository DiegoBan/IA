import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import itertools
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
def build_model(architecture, kernel_size):
    model = models.Sequential()

    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    for num_filters in architecture:
        model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax')) # 5 Clases
    return model

#   MAIN

param_grid = {  #   Parámetros a probar
    'batch_size': [128],
    'epochs': [15, 20],
    'learning_rate': [0.0001],
    'kernel_size': [(5, 5)],
    'architecture': [
        [32, 64],
        [32, 64, 128]
    ]
}

BASE_DIR = './img_divided'
keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

print(f"Total de combinaciones a probar: {len(combinations)}")

best_accuracy = 0.0
best_params = {}
best_model = None

for i, values in enumerate(tqdm(combinations, desc="Grid Search Progress", unit="config")):   
    current_params = dict(zip(keys, values))
    
    tqdm.write(f"\n--- Intento {i+1}/{len(combinations)} ---")
    tqdm.write(f"Probando: {current_params}")

    train_ds, val_ds = get_datasets(current_params['batch_size'])   #   Dataset a usar

    #   Creación de modelo
    model = build_model(
        architecture=current_params['architecture'],
        kernel_size=current_params['kernel_size']
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=current_params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(    #   Entrenamiento
        train_ds, 
        validation_data=val_ds, 
        epochs=current_params['epochs'],
        verbose=0
    )

    current_val_acc = max(history.history['val_accuracy'])
    
    tqdm.write(f"Result -> Val Accuracy: {current_val_acc:.4f}")
    
    if current_val_acc > best_accuracy:
        best_accuracy = current_val_acc
        best_params = current_params
        best_model = model
        tqdm.write("¡NUEVO MEJOR MODELO ENCONTRADO!")

    tf.keras.backend.clear_session()
    del model

#   Resultados finales
print("\n" + "="*50)
print("BÚSQUEDA FINALIZADA")
print(f"Mejor Val Accuracy: {best_accuracy:.4f}")
print("Mejores Hiperparámetros:")
for k, v in best_params.items():
    print(f"  - {k}: {v}")
print("="*50)