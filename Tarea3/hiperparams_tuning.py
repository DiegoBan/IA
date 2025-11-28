import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import itertools

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
    model.add(layers.Rescaling(1./255, input_shape=(64, 64, 3)))
    
    # Iteramos sobre la lista de arquitectura para crear las capas
    # Ejemplo: si architecture es [32, 64], crea 2 bloques Conv+Pool
    for num_filters in architecture:
        model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax')) # 5 Clases
    return model

param_grid = {  #   Parámetros a provar en sus distintas combinaciones
    'batch_size': [32, 64],
    'epochs': [10, 15],
    'learning_rate': [0.001, 0.0001],
    'kernel_size': [(3, 3), (5, 5)],
    'architecture': [   #   Cuantas capas y con cuantas neuronas
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

for i, values in enumerate(combinations):   #   Probar cada combinación
    current_params = dict(zip(keys, values))
    
    print(f"\n--- Intento {i+1}/{len(combinations)} ---")
    print(f"Probando: {current_params}")

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
    print(f"Result -> Val Accuracy: {current_val_acc:.4f}")
    if current_val_acc > best_accuracy:
        best_accuracy = current_val_acc
        best_params = current_params
        best_model = model
        model.save('best_model_temp.keras') #   Guarda mejor modelo en pc
        print("¡NUEVO MEJOR MODELO ENCONTRADO!")

#   Resultados finales
print("\n" + "="*50)
print("BÚSQUEDA FINALIZADA")
print(f"Mejor Val Accuracy: {best_accuracy:.4f}")
print("Mejores Hiperparámetros:")
for k, v in best_params.items():
    print(f"  - {k}: {v}")
print("="*50)