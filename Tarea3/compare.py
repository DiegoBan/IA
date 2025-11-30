import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def get_test_dataset(batch_size):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'testing_set'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        verbose=0,
        shuffle=False
    )
    return test_ds.cache().prefetch(tf.data.AUTOTUNE)

def get_predictions(model, test_ds):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        preds_classes = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds_classes)
    return np.array(y_true), np.array(y_pred)

def plot_history_comparison(hist_no_drop, hist_drop):
    epochs_range = range(1, len(hist_no_drop['accuracy']) + 1)
    
    plt.figure(figsize=(14, 6))

    # GRÁFICO 1: ACCURACY
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist_no_drop['accuracy'], label='Train (No Drop)', linestyle='--', color='blue', alpha=0.6)
    plt.plot(epochs_range, hist_no_drop['val_accuracy'], label='Val (No Drop)', color='blue')
    plt.plot(epochs_range, hist_drop['accuracy'], label='Train (Dropout)', linestyle='--', color='red', alpha=0.6)
    plt.plot(epochs_range, hist_drop['val_accuracy'], label='Val (Dropout)', color='red')
    plt.title('Comparación de Accuracy')
    plt.xlabel('Épocas'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    # GRÁFICO 2: LOSS
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist_no_drop['loss'], label='Train (No Drop)', linestyle='--', color='blue', alpha=0.6)
    plt.plot(epochs_range, hist_no_drop['val_loss'], label='Val (No Drop)', color='blue')
    plt.plot(epochs_range, hist_drop['loss'], label='Train (Dropout)', linestyle='--', color='red', alpha=0.6)
    plt.plot(epochs_range, hist_drop['val_loss'], label='Val (Dropout)', color='red')
    plt.title('Comparación de Loss')
    plt.xlabel('Épocas'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

#   MAIN

BASE_DIR = './img_divided'
LOAD_DIR = './final_models/'

print("Cargando modelos e historiales...")

#   Cargar modelos
model_no_drop = tf.keras.models.load_model(os.path.join(LOAD_DIR, 'modelo_no_dropout.keras'))
model_drop = tf.keras.models.load_model(os.path.join(LOAD_DIR, 'modelo_dropout.keras'))

#   Cargar historial
with open(os.path.join(LOAD_DIR, 'history_no_dropout.pkl'), 'rb') as f:
    hist_no_drop = pickle.load(f)
with open(os.path.join(LOAD_DIR, 'history_dropout.pkl'), 'rb') as f:
    hist_drop = pickle.load(f)

#   Obtener testing_set
test_ds = get_test_dataset(128)

#   Predicciones del modelo
print("Evaluando modelo Sin Dropout...")
y_true, y_pred_no_drop = get_predictions(model_no_drop, test_ds)

print("Evaluando modelo Con Dropout...")
_, y_pred_drop = get_predictions(model_drop, test_ds)

#   Calcular metricas
acc_no_drop = accuracy_score(y_true, y_pred_no_drop)
f1_no_drop = f1_score(y_true, y_pred_no_drop, average='weighted')
cm_no_drop = confusion_matrix(y_true, y_pred_no_drop)

acc_drop = accuracy_score(y_true, y_pred_drop)
f1_drop = f1_score(y_true, y_pred_drop, average='weighted')
cm_drop = confusion_matrix(y_true, y_pred_drop)

#   Tabla Comparativa
print("\n" + "="*60)
print(f"{'MÉTRICA':<20} | {'SIN DROPOUT':<15} | {'CON DROPOUT':<15}")
print("-" * 60)
print(f"{'Accuracy (Test)':<20} | {acc_no_drop:.4f}          | {acc_drop:.4f}")
print(f"{'F1-Score (Test)':<20} | {f1_no_drop:.4f}          | {f1_drop:.4f}")
print("="*60 + "\n")

#   Graficar Curvas
plot_history_comparison(hist_no_drop, hist_drop)

#   Graficar Matrices de Confusión
class_names = ["AnnualCrop", "Forest", "Herbaceous", "Residential", "SeaLake"] 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm_no_drop, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('Matriz Confusión: SIN Dropout')
axes[0].set_ylabel('Realidad'); axes[0].set_xlabel('Predicción')

sns.heatmap(cm_drop, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Matriz Confusión: CON Dropout')
axes[1].set_ylabel('Realidad'); axes[1].set_xlabel('Predicción')

plt.tight_layout()
plt.show()