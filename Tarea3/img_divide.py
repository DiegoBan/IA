import os
import shutil
import random

source_dir = './img_total'  #   Ruta Origen
output_dir = './img_divided'    #   Ruta Destino
sets = ['training_set', 'validation_set', 'testing_set']    #   Division
classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Residential", "SeaLake"]    #   Clases existentes
# Porcentajes de división
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

if (train_ratio + val_ratio + test_ratio) != 1.0:
    raise ValueError("porcentajes deben sumar 1")
print("==== Iniciando division ====")
for cls in classes:
    cls_path = os.path.join(source_dir, cls)    #   Ruta clase de origen
    if not os.path.exists(cls_path):
        raise ValueError(f"Ruta faltante {cls_path}")
    #   Lista de archivos y randomizar
    files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    random.shuffle(files)
    total_files = len(files)
    print(f"Procesando {cls}: {total_files} img encontradas")
    #   Calcular cantidades
    train_count = int(total_files*train_ratio)
    val_count = int(total_files*val_ratio)
    # El resto va para test, asi no se pierde nada por redondeo usando int
    #   Se dividen los archivos
    train_files = files[:train_count]
    val_files = files[train_count:val_count + train_count]
    test_files = files[val_count + train_count :]
    #   Función para copiar
    def copy_files(file_list, set_name):
        for file_name in file_list:
            src = os.path.join(cls_path, file_name)
            dest_folder = os.path.join(output_dir, set_name, cls)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy2(src, os.path.join(dest_folder, file_name))
    #   Copia para cada grupo
    copy_files(train_files, 'training_set')
    copy_files(val_files, 'validation_set')
    copy_files(test_files, 'testing_files')
    print(f"-> {len(train_files)} a Train, -> {len(val_files)} a Validation, -> {len(test_files)} a Test")
print("Proceso terminado")