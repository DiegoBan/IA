import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from tqdm import tqdm  # <--- IMPORTANTE: Importamos la librería

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

BASE_DIR = './img_divided'

#   Valores a probar
batch_size = {
    'values': [16, 32, 64, 128],
    'accuracy': [],
    'best_index': 0
}
epochs = {
    'values': [5, 10, 15, 20],
    'accuracy': [],
    'best_index': 0
}
learning_rate = {
    'values': [0.001, 0.1, 0.01, 0.0001],
    'accuracy': [],
    'best_index': 0
}
kernel_size = {
    'values': [(3, 3), (2, 2), (5, 5), (8, 8)],
    'accuracy': [],
    'best_index': 0
}
architecture = {
    'values': [
        [32, 64],
        [32],
        [32, 64, 128],
        [32, 64, 128, 256]
    ],
    'accuracy': [],
    'best_index': 0
}

print("Iniciando prueba de parametros")

for i, val in enumerate(tqdm(batch_size['values'], desc="Batch Size")):
    tqdm.write(f"Probando batch size = {val}")
    train_ds, val_ds = get_datasets(val)
    
    model = build_model(
        architecture=architecture['values'][architecture['best_index']],
        kernel_size=kernel_size['values'][kernel_size['best_index']]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate['values'][learning_rate['best_index']]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs['values'][epochs['best_index']],
        verbose=0
    )
    
    accuracy = max(history.history['val_accuracy'])
    batch_size['accuracy'].append(accuracy)
    if accuracy > batch_size['accuracy'][batch_size['best_index']]:
        batch_size['best_index'] = i
    tf.keras.backend.clear_session()
    del model

for i, val in enumerate(tqdm(epochs['values'], desc="Epochs")):
    tqdm.write(f"Probando epoch = {val}")
    train_ds, val_ds = get_datasets(batch_size['values'][batch_size['best_index']])
    
    model = build_model(
        architecture=architecture['values'][architecture['best_index']],
        kernel_size=kernel_size['values'][kernel_size['best_index']]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate['values'][learning_rate['best_index']]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=val,
        verbose=0
    )
    
    accuracy = max(history.history['val_accuracy'])
    epochs['accuracy'].append(accuracy)
    if accuracy > epochs['accuracy'][epochs['best_index']]:
        epochs['best_index'] = i
    tf.keras.backend.clear_session()
    del model

for i, val in enumerate(tqdm(learning_rate['values'], desc="Learning rate")):
    tqdm.write(f"Probando learning rate = {val}")
    train_ds, val_ds = get_datasets(batch_size['values'][batch_size['best_index']])
    
    model = build_model(
        architecture=architecture['values'][architecture['best_index']],
        kernel_size=kernel_size['values'][kernel_size['best_index']]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=val),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs['values'][epochs['best_index']],
        verbose=0
    )
    
    accuracy = max(history.history['val_accuracy'])
    learning_rate['accuracy'].append(accuracy)
    if accuracy > learning_rate['accuracy'][learning_rate['best_index']]:
        learning_rate['best_index'] = i
    tf.keras.backend.clear_session()
    del model

for i, val in enumerate(tqdm(kernel_size['values'], desc="Kernel size")):
    tqdm.write(f"Probando kernel size = {val}")
    train_ds, val_ds = get_datasets(batch_size['values'][batch_size['best_index']])
    
    model = build_model(
        architecture=architecture['values'][architecture['best_index']],
        kernel_size=val
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate['values'][learning_rate['best_index']]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs['values'][epochs['best_index']],
        verbose=0
    )
    
    accuracy = max(history.history['val_accuracy'])
    kernel_size['accuracy'].append(accuracy)
    if accuracy > kernel_size['accuracy'][kernel_size['best_index']]:
        kernel_size['best_index'] = i
    tf.keras.backend.clear_session()
    del model

for i, val in enumerate(tqdm(architecture['values'], desc="Architecture")):
    tqdm.write(f"Probando architecture = {val}")
    train_ds, val_ds = get_datasets(batch_size['values'][batch_size['best_index']])
    
    model = build_model(
        architecture=val,
        kernel_size=kernel_size['values'][kernel_size['best_index']]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate['values'][learning_rate['best_index']]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs['values'][epochs['best_index']],
        verbose=0
    )
    
    accuracy = max(history.history['val_accuracy'])
    architecture['accuracy'].append(accuracy)
    if accuracy > architecture['accuracy'][architecture['best_index']]:
        architecture['best_index'] = i
    tf.keras.backend.clear_session()
    del model

#   Imprimir resultados finales
print("\n" + "="*30)
print(" RESULTADOS FINALES ")
print("="*30 + "\n")

#   Batch size
print("="*10, " Batch size ", "="*10)
for i, val in enumerate(batch_size['values']):
    print(f"{i}. batch size = {val}, Accuracy = {batch_size['accuracy'][i]:.4f}")

#   Epoch
print("="*10, " Epochs ", "="*10)
for i, val in enumerate(epochs['values']):
    print(f"{i}. Epochs = {val}, Accuracy = {epochs['accuracy'][i]:.4f}")

#   Learning Rate
print("="*10, " Learning Rate ", "="*10)
for i, val in enumerate(learning_rate['values']):
    print(f"{i}. learning rate = {val}, Accuracy = {learning_rate['accuracy'][i]:.4f}")

#   Kernel size
print("="*10, " Kernel Size ", "="*10)
for i, val in enumerate(kernel_size['values']):
    print(f"{i}. kernel size = {val}, Accuracy = {kernel_size['accuracy'][i]:.4f}")

#   Architecture
print("="*10, " Architecture ", "="*10)
for i, val in enumerate(architecture['values']):
    print(f"{i}. architecture = {val}, Accuracy = {architecture['accuracy'][i]:.4f}")