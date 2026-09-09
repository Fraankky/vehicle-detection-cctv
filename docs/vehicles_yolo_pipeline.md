# Panduan `vehicles_yolo_pipeline.ipynb`

Dokumen ini menjelaskan setiap cell pada `vehicles_yolo_pipeline.ipynb`, termasuk fungsi setiap baris penting dan urutan eksekusinya.

## Urutan menjalankan notebook

1. Jalankan cell instalasi dependency.
2. Jalankan cell import dan konfigurasi.
3. Masukkan `ROBOFLOW_API_KEY` saat diminta, lalu download dataset.
4. Jalankan verifikasi dataset, visualisasi anotasi, dan EDA.
5. Di Colab dengan GPU, jalankan training baseline.
6. Evaluasi baseline pada validation split.
7. Jalankan eksperimen improvement jika diperlukan.
8. Bandingkan hasil validasi, lalu pilih model final.
9. Upload `best.pt` jika model berasal dari sesi Colab lain.
10. Jalankan evaluasi test, error analysis, inference gambar/video, dan report.

`valid` dipakai untuk memilih model. `test` hanya dipakai untuk evaluasi final agar hasil test tidak bocor ke proses tuning.

## Cell 1: Overview

### Jenis

Markdown.

### Fungsi

Menjelaskan tujuan notebook, urutan proses, aturan keamanan API key, dan catatan bahwa training sebaiknya dijalankan di Colab dengan GPU.

### Hal penting

- Dataset yang digunakan adalah Vehicles dari Roboflow 100.
- Format dataset adalah YOLOv8.
- Hasil akhir yang diharapkan adalah `best.pt`, metrik evaluasi, visualisasi error, dan report.
- API key tidak ditulis langsung di notebook.

## Cell 2: Install dependency

```python
%pip install -q ultralytics roboflow opencv-python matplotlib seaborn pandas pyyaml scikit-learn pillow
```

### Penjelasan per baris

- `%pip install`: menjalankan `pip` dari kernel Jupyter/Colab yang sedang aktif.
- `-q`: quiet mode, mengurangi output instalasi agar notebook lebih rapi.
- `ultralytics`: library YOLO untuk training, validasi, dan inference.
- `roboflow`: SDK untuk mengakses dan mendownload dataset Roboflow.
- `opencv-python`: membaca gambar/video, menggambar bounding box, dan mengukur inference video.
- `matplotlib`: membuat plot dan menampilkan gambar.
- `seaborn`: membuat visualisasi statistik yang lebih rapi.
- `pandas`: menyimpan dan mengolah tabel hasil EDA/evaluasi.
- `pyyaml`: membaca file konfigurasi `data.yaml`.
- `scikit-learn`: disiapkan untuk kebutuhan analisis metrik tambahan.
- `pillow`: dukungan operasi gambar yang umum digunakan oleh ekosistem Python.

Cell ini cukup dijalankan sekali per runtime. Setelah restart runtime, dependency mungkin perlu di-install lagi.

## Cell 3: Import dan konfigurasi dasar

```python
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import shutil
import time
from collections import Counter, defaultdict
from getpass import getpass
```

### Penjelasan

- `Path`: memanipulasi path file secara lintas platform.
- `hashlib`: membuat hash file untuk mendeteksi gambar duplikat.
- `json`: menyimpan informasi training dalam `training_info.json`.
- `os`: membaca environment variable seperti `ROBOFLOW_API_KEY`.
- `platform`: membaca versi Python.
- `random`: memilih sampel gambar secara acak tetapi reproducible.
- `shutil`: menyalin file `best.pt` dari hasil upload.
- `time`: mengukur durasi training dan waktu inference.
- `Counter`, `defaultdict`: utilitas penghitung; disiapkan untuk analisis data.
- `getpass`: meminta API key tanpa menampilkannya di layar.

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

from IPython.display import Image as IPImage, display
```

- `cv2`: OpenCV untuk membaca gambar/video dan menggambar bounding box.
- `plt`: API plotting dari Matplotlib.
- `np`: operasi array numerik.
- `pd`: tabel dan agregasi data melalui Pandas.
- `sns`: visualisasi statistik melalui Seaborn.
- `torch`: mengecek ketersediaan CUDA/GPU.
- `yaml`: membaca isi `data.yaml`.
- `IPImage`: menampilkan file PNG hasil evaluasi di notebook.
- `display`: menampilkan object atau DataFrame dengan format notebook.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style='whitegrid')
```

- `SEED = 42`: seed tetap agar pemilihan sampel dan operasi acak dapat diulang.
- `random.seed(SEED)`: mengatur random generator Python.
- `np.random.seed(SEED)`: mengatur random generator NumPy.
- `sns.set_theme(...)`: memberi gaya grid pada plot Seaborn.

```python
PROJECT_ROOT = Path.cwd()
DATASET_ROOT = PROJECT_ROOT / 'datasets' / 'vehicles'
RUNS_ROOT = PROJECT_ROOT / 'runs'
WEIGHTS_ROOT = PROJECT_ROOT / 'weights'
REPORTS_ROOT = PROJECT_ROOT / 'reports'
```

- `PROJECT_ROOT`: folder aktif tempat notebook dijalankan.
- `DATASET_ROOT`: lokasi dataset hasil download.
- `RUNS_ROOT`: lokasi hasil training, evaluasi, dan inference.
- `WEIGHTS_ROOT`: lokasi model final seperti `best.pt`.
- `REPORTS_ROOT`: lokasi report Markdown.

```python
for directory in (DATASET_ROOT, RUNS_ROOT, WEIGHTS_ROOT, REPORTS_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
```

- Loop ini melewati seluruh folder output.
- `mkdir(..., parents=True)`: membuat folder induk jika belum ada.
- `exist_ok=True`: tidak error jika folder sudah ada.

```python
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f'Project root: {PROJECT_ROOT}')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
```

- `torch.cuda.is_available()`: mengecek apakah CUDA tersedia.
- `DEVICE = 0`: meminta Ultralytics memakai GPU pertama.
- `DEVICE = 'cpu'`: fallback jika GPU tidak tersedia.
- Dua `print`: menampilkan lokasi project dan device.
- `get_device_name(0)`: menampilkan nama GPU pertama.

## Cell 4: Judul download dataset

### Jenis

Markdown.

### Fungsi

Menjelaskan mengapa format `yolov8` digunakan dan di mana dataset akan disimpan. Format ini langsung dikenali Ultralytics dan menghasilkan `data.yaml`.

## Cell 5: Download dataset Roboflow

```python
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
if not ROBOFLOW_API_KEY:
    ROBOFLOW_API_KEY = getpass('Masukkan Roboflow API key: ')
if not ROBOFLOW_API_KEY:
    raise ValueError('ROBOFLOW_API_KEY kosong.')
```

- `os.getenv(...)`: mengambil key dari environment variable.
- Jika variable belum ada, `getpass(...)` meminta key secara tersembunyi.
- `raise ValueError`: menghentikan notebook jika key kosong.
- Key tidak ditulis permanen di source code.

```python
from roboflow import Roboflow

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace('muhammad-faris').project('vehicles-q0x2v-kpycp')
version = project.version(1)
dataset = version.download('yolov8')
```

- Import `Roboflow` dari SDK.
- Membuat client Roboflow dengan API key.
- Memilih workspace `muhammad-faris`.
- Memilih project `vehicles-q0x2v-kpycp`.
- Memilih version dataset nomor `1`.
- Download dataset dalam format YOLOv8 ke lokasi default yang dipilih SDK Roboflow.

```python
DATASET_ROOT = Path(dataset.location)
DATA_YAML = DATASET_ROOT / 'data.yaml'
print(f'Dataset location: {DATASET_ROOT}')
print(f'Data YAML: {DATA_YAML}')
assert DATA_YAML.exists(), f'data.yaml tidak ditemukan: {DATA_YAML}'
```

- `dataset.location`: lokasi aktual yang dikembalikan SDK.
- `DATASET_ROOT`: diperbarui memakai lokasi aktual tersebut.
- `DATA_YAML`: path konfigurasi dataset.
- `print`: menampilkan path untuk debugging.
- `assert`: memastikan download menghasilkan `data.yaml`.

> Jika dataset sudah ada, SDK dapat mengeluarkan pesan folder sudah tersedia. Pastikan isi folder memang berasal dari version yang benar.

> Pada pola download standar `dataset = version.download('yolov8')`, `DATASET_ROOT` dan `DATA_YAML` tetap wajib dibuat dari `dataset.location` sebelum menjalankan cell pembacaan YAML. Tanpa dua assignment tersebut, cell berikutnya akan menghasilkan `NameError: DATA_YAML is not defined`.

## Cell 6: Baca konfigurasi dan daftar gambar

```python
DATASET_ROOT = Path(dataset.location)
DATA_YAML = DATASET_ROOT / 'data.yaml'

with DATA_YAML.open() as file:
    DATA_CONFIG = yaml.safe_load(file)
```

- `DATASET_ROOT`: mengambil lokasi dataset langsung dari object `dataset` hasil download.
- `DATA_YAML`: menunjuk file konfigurasi dataset.
- Membuka `data.yaml`.
- `yaml.safe_load`: mengubah YAML menjadi dictionary Python.
- `DATA_CONFIG` biasanya berisi `path`, `train`, `val`, `test`, `names`, dan `nc`.

```python
raw_names = DATA_CONFIG.get('names', [])
if isinstance(raw_names, dict):
    CLASS_NAMES = [raw_names[key] for key in sorted(raw_names, key=lambda value: int(value))]
else:
    CLASS_NAMES = list(raw_names)
```

- Mengambil daftar nama kelas.
- Roboflow/Ultralytics dapat menulis `names` sebagai dictionary atau list.
- Jika dictionary, key diurutkan berdasarkan ID numerik agar `CLASS_NAMES[0]` tetap kelas ID 0.
- Jika list, langsung dikonversi menjadi list Python.

```python
def image_paths(split):
    key = 'val' if split == 'valid' else split
    value = DATA_CONFIG.get(key)
    if value is None:
        return []
    path = Path(value)
    if not path.is_absolute():
        path = DATASET_ROOT / path
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    if path.is_dir():
        return sorted(item for item in path.rglob('*') if item.suffix.lower() in extensions)
    if path.is_file() and path.suffix.lower() in extensions:
        return [path]
    return sorted(item for item in path.parent.glob(path.name) if item.suffix.lower() in extensions)
```

- `key`: memakai `val`, nama split standar Ultralytics.
- `value`: mengambil path split dari `data.yaml`.
- Path relatif digabung langsung dengan `DATASET_ROOT`.
- Folder dicari secara recursive, file tunggal dikembalikan langsung, dan wildcard dicari dengan `glob`.

```python
SPLIT_IMAGES = {split: image_paths(split) for split in ('train', 'valid', 'test')}
print('Classes:', CLASS_NAMES)
print('Jumlah kelas:', len(CLASS_NAMES))
for split, paths in SPLIT_IMAGES.items():
    print(f'{split:>5}: {len(paths):>4} gambar')

assert CLASS_NAMES, 'Daftar kelas kosong.'
assert SPLIT_IMAGES['train'], 'Gambar train tidak ditemukan.'
```

- Membuat dictionary berisi path gambar untuk train, valid, dan test.
- Mencetak nama serta jumlah kelas.
- Mencetak jumlah gambar tiap split.
- `assert CLASS_NAMES`: gagal jika daftar kelas tidak ditemukan.
- `assert SPLIT_IMAGES['train']`: gagal jika gambar training tidak ditemukan.

## Cell 7: Judul verifikasi anotasi

### Jenis

Markdown.

### Fungsi

Menjelaskan alasan visualisasi anotasi dilakukan sebelum training: menemukan label rusak, koordinat tidak valid, atau class ID yang tidak sesuai lebih awal.

## Cell 8: Load label dan visualisasi sample

```python
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if 'images' in parts:
        parts[parts.index('images')] = 'labels'
        return Path(*parts).with_suffix('.txt')
    return image_path.with_suffix('.txt')
```

- Mengubah path gambar menjadi path label YOLO.
- Pada struktur standar, `images/foo.jpg` dipasangkan dengan `labels/foo.txt`.
- Jika folder `images` tidak ada di path, cukup ganti ekstensi gambar menjadi `.txt`.

```python
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    labels = []
```

- Mendapatkan path label.
- Jika label tidak ada, mengembalikan list kosong.
- `labels` menampung anotasi yang valid.

```python
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        try:
            numbers = [float(value) for value in values]
        except ValueError:
            continue
        if not numbers[0].is_integer():
            continue
        labels.append((int(numbers[0]), *numbers[1:]))
    return labels
```

- Membaca label baris demi baris.
- Format YOLO object detection harus memiliki lima nilai: `class_id x_center y_center width height`.
- Baris dengan jumlah nilai salah dilewati.
- Nilai dikonversi menjadi float.
- Class ID harus berupa bilangan bulat.
- Hasil dikembalikan sebagai tuple `(class_id, x_center, y_center, box_width, box_height)`.
- Koordinat YOLO masih normalized pada rentang 0 sampai 1.

```python
    paths = SPLIT_IMAGES[split]
    if not paths:
        print(f'Tidak ada gambar pada split {split}.')
        return
```

- Mendefinisikan fungsi visualisasi sample.
- Default mengambil data train dan maksimal 12 gambar.
- Jika split kosong, fungsi berhenti dengan pesan.

```python
    samples = random.Random(SEED).sample(paths, min(count, len(paths)))
    columns = 3
    rows = int(np.ceil(len(samples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 5 * rows))
    axes = np.atleast_1d(axes).ravel()
```

- Memilih sample reproducible.
- Membuat layout tiga kolom.
- Menghitung jumlah baris.
- Membuat figure Matplotlib.
- Memastikan `axes` selalu bisa di-loop walaupun hanya ada satu subplot.

```python
    for axis, image_path in zip(axes, samples):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
```

- Loop setiap gambar sample.
- OpenCV membaca gambar dalam format BGR.
- Konversi ke RGB agar warna benar saat ditampilkan Matplotlib.
- Mengambil tinggi dan lebar gambar.

```python
        for class_id, x_center, y_center, box_width, box_height in load_yolo_labels(image_path):
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)
```

- Loop setiap anotasi pada gambar.
- Mengubah format normalized YOLO `xywh` menjadi pixel corner coordinates `x1, y1, x2, y2`.

```python
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f'class_{class_id}'
            cv2.putText(image, label, (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 60, 0), 2)
```

- Menggambar bounding box hijau.
- Mengambil nama kelas dari class ID.
- Jika class ID tidak valid, memakai nama fallback.
- Menulis nama kelas di atas bounding box.

```python
        axis.imshow(image)
        axis.set_title(image_path.name)
        axis.axis('off')
```

- Menampilkan gambar pada subplot.
- Memberi judul berupa nama file.
- Menyembunyikan sumbu koordinat.

```python
    for axis in axes[len(samples):]:
        axis.axis('off')
    figure.suptitle(f'Sampel anotasi: {split}', fontsize=16)
    figure.tight_layout()
    plt.show()

plot_samples('train', count=12)
```

- Mematikan subplot kosong jika jumlah sample tidak memenuhi seluruh grid.
- Memberi judul figure.
- Merapikan layout dan menampilkan figure.
- Memanggil fungsi untuk menampilkan 12 gambar train.

## Cell 9: Validasi label

```python
    invalid = []
    missing = []
    empty = []
```

- Mendefinisikan validasi untuk satu split.
- `invalid`: label dengan format atau nilai salah.
- `missing`: gambar yang tidak memiliki file label.
- `empty`: file label yang ada tetapi kosong.

```python
    for image_path in SPLIT_IMAGES[split]:
        label_path = label_path_for_image(image_path)
        if not label_path.exists():
            missing.append(str(image_path))
            continue
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            empty.append(str(image_path))
```

- Memeriksa setiap gambar pada split.
- Mencari file label pasangannya.
- Mencatat gambar tanpa label.
- Menghapus baris kosong.
- Mencatat label file yang tidak memiliki anotasi.

```python
        for line_number, line in enumerate(lines, start=1):
            values = line.split()
            try:
                numbers = [float(value) for value in values]
                valid = (len(numbers) == 5 and numbers[0].is_integer() and 0 <= int(numbers[0]) < len(CLASS_NAMES) and all(0 <= value <= 1 for value in numbers[1:]))
            except ValueError:
                valid = False
            if not valid:
                invalid.append((str(label_path), line_number, line))
```

- Menyimpan nomor baris agar error mudah dilacak.
- Memastikan ada tepat lima nilai.
- Memastikan class ID integer dan berada dalam range kelas.
- Memastikan empat koordinat normalized berada pada rentang 0 sampai 1.
- Jika konversi gagal atau syarat tidak terpenuhi, anotasi dicatat sebagai invalid.

```python
    return {'missing': missing, 'empty': empty, 'invalid': invalid}

validation = {split: validate_labels(split) for split in SPLIT_IMAGES}
for split, result in validation.items():
    print(f'{split}: missing={len(result["missing"])}, empty={len(result["empty"])}, invalid={len(result["invalid"])}')
    if result['invalid']:
        print('Contoh label invalid:', result['invalid'][:3])
```

- Mengembalikan ringkasan tiga jenis masalah.
- Menjalankan validasi untuk train, valid, dan test.
- Mencetak jumlah masalah per split.
- Menampilkan maksimal tiga contoh label invalid.

## Cell 10: Judul EDA

### Jenis

Markdown.

### Fungsi

Menjelaskan bahwa EDA dipakai untuk mengetahui class imbalance, ukuran objek, dan kemungkinan gambar duplikat atau leakage antar split.

## Cell 11: Distribusi kelas dan ukuran bounding box

```python
    if area_ratio < 0.01:
        return 'small (<1%)'
    if area_ratio < 0.10:
        return 'medium (1-10%)'
    return 'large (>=10%)'
```

- Membuat kategori ukuran objek berdasarkan rasio area bounding box terhadap area gambar.
- `small`: kurang dari 1% area gambar.
- `medium`: 1% sampai kurang dari 10%.
- `large`: minimal 10%.
- Ini adalah heuristic untuk EDA, bukan definisi resmi COCO.

```python
eda_rows = []
for split, paths in SPLIT_IMAGES.items():
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
```

- Menyiapkan list baris EDA.
- Loop semua split dan gambar.
- Membaca gambar untuk mengambil dimensinya.
- Gambar yang gagal dibaca dilewati.

```python
        for class_id, x_center, y_center, box_width, box_height in load_yolo_labels(image_path):
            area_ratio = box_width * box_height
            eda_rows.append({
                'split': split,
                'image': str(image_path),
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f'class_{class_id}',
                'width_px': width,
                'height_px': height,
                'box_width_ratio': box_width,
                'box_height_ratio': box_height,
                'area_ratio': area_ratio,
                'size': size_bucket(area_ratio),
            })
```

- Loop seluruh object pada gambar.
- Menghitung rasio area box karena koordinat width dan height sudah normalized.
- Menyimpan metadata object sebagai dictionary.
- Menyimpan class, ukuran gambar, ukuran box, rasio area, dan bucket ukuran.

```python
eda_df = pd.DataFrame(eda_rows)
if eda_df.empty:
    raise ValueError('Tidak ada instance anotasi yang terbaca.')
```

- Mengubah list object menjadi DataFrame.
- Menghentikan proses jika tidak ada anotasi yang terbaca.

```python
print('Distribusi instance per kelas:')
display(eda_df.groupby(['split', 'class_name']).size().unstack(fill_value=0))
print('Distribusi ukuran objek:')
display(eda_df.groupby(['split', 'size']).size().unstack(fill_value=0))
```

- Mengelompokkan jumlah object berdasarkan split dan kelas.
- Mengelompokkan jumlah object berdasarkan split dan kategori ukuran.
- `unstack` membuat hasil lebih mudah dibaca sebagai tabel.

```python
figure, axes = plt.subplots(1, 2, figsize=(17, 5))
class_counts = eda_df.groupby(['split', 'class_name']).size().unstack(fill_value=0)
class_counts.plot(kind='bar', ax=axes[0])
axes[0].set_title('Instance per kelas dan split')
axes[0].set_xlabel('Split')
axes[0].set_ylabel('Jumlah instance')
axes[0].tick_params(axis='x', rotation=0)
```

- Membuat dua subplot.
- Mengubah distribusi kelas menjadi bar chart.
- Memberi judul dan label sumbu.

```python
sns.histplot(data=eda_df[eda_df['split'] == 'train'], x='area_ratio', hue='class_name', bins=30, element='step', stat='count', ax=axes[1])
axes[1].set_title('Distribusi area bounding box train')
axes[1].set_xlabel('Area box / area gambar')
figure.tight_layout()
plt.show()
```

- Membuat histogram ukuran box hanya dari train.
- `hue='class_name'` membedakan distribusi antar kelas.
- Menampilkan apakah objek kecil mendominasi dataset.

## Cell 12: Deteksi duplikat dan leakage

```python
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return ''.join('1' if pixel >= image.mean() else '0' for pixel in image.ravel())
```

- Membaca gambar dalam grayscale.
- Gambar invalid menghasilkan `None`.
- Resize menjadi 8x8 agar gambar bisa dibandingkan secara kasar.
- Setiap pixel dibandingkan dengan rata-rata pixel.
- Hasil binary string disebut average hash.
- Hash yang sama mengindikasikan gambar sangat mirip, tetapi bukan bukti pasti duplikat.

```python
for split, paths in SPLIT_IMAGES.items():
    for image_path in paths:
        content = image_path.read_bytes()
        duplicate_rows.append({
            'split': split,
            'image': str(image_path),
            'sha256': hashlib.sha256(content).hexdigest(),
            'average_hash': average_hash(image_path),
        })
```

- Membaca isi byte setiap file.
- SHA-256 mendeteksi exact duplicate.
- Average hash mendeteksi kemiripan visual kasar.
- Semua informasi disimpan sebagai DataFrame.

```python
    groups = duplicate_df.groupby(column).agg(
        count=('image', 'size'),
        splits=('split', lambda values: ','.join(sorted(set(values)))),
        images=('image', list),
    )
    return groups[groups['count'] > 1].sort_values('count', ascending=False)
```

- Mengelompokkan gambar berdasarkan kolom hash.
- Menghitung jumlah gambar per kelompok.
- Mencatat split yang terlibat.
- Menyimpan daftar path gambar.
- Hanya mengembalikan kelompok yang berisi lebih dari satu gambar.

```python
exact_duplicates = duplicate_groups('sha256')
similar_duplicates = duplicate_groups('average_hash')
print(f'Exact duplicate groups: {len(exact_duplicates)}')
if not exact_duplicates.empty:
    display(exact_duplicates.head(10))
print(f'Perceptual duplicate groups: {len(similar_duplicates)}')
if not similar_duplicates.empty:
    display(similar_duplicates.head(10))
```

- Mencari exact duplicate dan perceptual duplicate.
- Menampilkan maksimal sepuluh kelompok terbesar.

```python
cross_split_hashes = duplicate_df.groupby('sha256')['split'].nunique()
leakage_candidates = cross_split_hashes[cross_split_hashes > 1]
print(f'Exact duplicate candidates across split: {len(leakage_candidates)}')
if len(leakage_candidates):
    display(duplicate_df[duplicate_df['sha256'].isin(leakage_candidates.index)])
```

- Menghitung berapa split berbeda yang memiliki SHA-256 sama.
- Jika satu hash muncul di lebih dari satu split, itu kandidat data leakage.
- Data leakage dapat membuat metrik validasi/test terlihat terlalu bagus.

## Cell 13: Penjelasan baseline training

### Jenis

Markdown.

### Fungsi

Menjelaskan parameter training baseline, early stopping, dan augmentasi default Ultralytics.

- Model baseline: `yolov8s.pt`.
- Ukuran gambar: `640`.
- Epoch maksimum: `100`.
- Batch: `16`.
- Patience: `20`.
- Augmentasi default membantu mencegah overfitting pada dataset kecil.
- Jika GPU kehabisan memori, batch dapat diturunkan menjadi 8 atau 4.

## Cell 14: Fungsi training baseline

```python
import ultralytics
from ultralytics import YOLO
```

- Mengimpor package untuk mencatat versinya.
- Mengimpor class `YOLO` untuk training.

```python
    model = YOLO(model_name)
    started = time.perf_counter()
```

- Mendefinisikan fungsi training yang dapat dipakai baseline dan improvement.
- `YOLO(model_name)`: memuat pretrained weights, misalnya `yolov8s.pt`.
- `perf_counter`: clock presisi tinggi untuk durasi training.

```python
    results = model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=epochs,
        batch=batch,
        patience=20,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=True,
        plots=True,
    )
```

- `data`: path ke `data.yaml`.
- `imgsz=640`: semua input training disiapkan pada resolusi 640.
- `epochs`: jumlah maksimum putaran training.
- `batch`: jumlah gambar per batch.
- `patience=20`: early stopping setelah 20 epoch tanpa perbaikan validasi.
- `device`: GPU pertama atau CPU.
- `project`: folder utama output Ultralytics.
- `name`: nama subfolder run.
- `exist_ok=True`: izinkan folder run sudah ada.
- `plots=True`: simpan grafik training dan evaluasi bawaan.

```python
    elapsed = time.perf_counter() - started
    save_dir = getattr(results, 'save_dir', None) or getattr(getattr(model, 'trainer', None), 'save_dir', None)
    run_dir = Path(save_dir or RUNS_ROOT / run_name)
    best_path = run_dir / 'weights' / 'best.pt'
```

- Menghitung durasi training.
- Mengambil lokasi output dari object hasil atau trainer.
- Fallback ke `runs/<run_name>` jika lokasi tidak tersedia.
- Menentukan path model terbaik.

```python
    info = {
        'model_name': model_name,
        'run_name': run_name,
        'elapsed_seconds': elapsed,
        'device': str(DEVICE),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(),
        'ultralytics': ultralytics.__version__,
        'imgsz': 640,
        'epochs': epochs,
        'batch': batch,
        'patience': 20,
    }
```

- Mengumpulkan metadata reproducibility.
- Mencatat model, durasi, hardware, versi Python, versi Ultralytics, dan parameter training.

```python
    (run_dir / 'training_info.json').write_text(json.dumps(info, indent=2))
    print(f'Training selesai dalam {elapsed / 60:.1f} menit')
    print(f'Run directory: {run_dir}')
    print(f'Best weights: {best_path}')
    return model, run_dir, best_path, info
```

- Menyimpan metadata sebagai JSON.
- Mencetak lokasi hasil.
- Mengembalikan model, folder run, path `best.pt`, dan metadata.

```python
BASELINE_MODEL_NAME = 'yolov8s.pt'
BASELINE_RUN_NAME = 'vehicles_yolov8s_baseline'

baseline_model, baseline_run_dir, BASELINE_BEST, BASELINE_INFO = train_model(
    BASELINE_MODEL_NAME, BASELINE_RUN_NAME, batch=16, epochs=100
)
```

- Menetapkan konfigurasi baseline.
- Memanggil training.
- Cell ini memang memulai training dan dapat memerlukan waktu lama.

## Cell 15: Penjelasan helper evaluasi

### Jenis

Markdown.

### Fungsi

Menjelaskan bahwa validation dipakai untuk pemilihan model, test untuk angka final, dan confidence rendah dipakai agar PR curve memiliki kandidat prediksi yang cukup.

## Cell 16: Helper metrik dan evaluasi

### `_as_array`

```python
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
```

- Mengubah berbagai bentuk output Ultralytics menjadi array 1D.
- Jika atribut kosong atau tidak bisa dikonversi, mengembalikan array kosong.

### `per_class_metric`

```python
    values = _as_array(getattr(box_metrics, attribute, []))
    output = np.full(len(CLASS_NAMES), np.nan)
    if len(values) == len(CLASS_NAMES):
        return values
```

- Mengambil atribut metrik seperti precision `p` atau recall `r`.
- Membuat output dengan satu slot per kelas.
- Jika output sudah memiliki panjang yang sama dengan jumlah kelas, langsung digunakan.

```python
    indices = _as_array(getattr(box_metrics, 'ap_class_index', []))
    for class_index, value in zip(indices.astype(int), values):
        if 0 <= class_index < len(output):
            output[class_index] = value
    return output
```

- Beberapa versi Ultralytics hanya mengembalikan metrik untuk kelas yang memiliki anotasi.
- `ap_class_index` memetakan metrik kembali ke class ID asli.
- Kelas yang tidak memiliki data tetap `NaN`.

### `per_class_ap`

```python
    output = np.full(len(CLASS_NAMES), np.nan)
    all_ap = np.asarray(getattr(box_metrics, 'all_ap', []), dtype=float)
    indices = _as_array(getattr(box_metrics, 'ap_class_index', []))
```

- Menyiapkan output AP per kelas.
- `all_ap` biasanya berisi AP setiap kelas pada beberapa IoU.

```python
    if all_ap.ndim == 2 and all_ap.shape[1] > column:
        for class_index, value in zip(indices.astype(int), all_ap[:, column]):
            if 0 <= class_index < len(output):
                output[class_index] = value
    return output
```

- Memastikan `all_ap` berbentuk matrix dan kolom tersedia.
- Mengambil AP pada kolom IoU yang diminta.
- Mengembalikannya ke posisi class ID yang benar.

### `metrics_table`

```python
    box = metrics.box
    precision = per_class_metric(box, 'p')
    recall = per_class_metric(box, 'r')
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
```

- Mengambil object detection metrics.
- Mengambil precision dan recall per kelas.
- Menghitung F1 per kelas dengan rumus `2PR / (P+R)`.
- `where` mencegah pembagian dengan nol.

```python
    return pd.DataFrame({
        'model': model_label,
        'split': split,
        'class': CLASS_NAMES,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': per_class_ap(box, 0),
        'mAP50-95': per_class_ap(box, 0 if np.asarray(getattr(box, 'all_ap', [])).ndim < 2 else np.asarray(box.all_ap).shape[1] - 1),
    })
```

- Membuat tabel per kelas.
- `mAP50`: memakai AP pada IoU 0.50.
- `mAP50-95`: memakai nilai agregat AP yang tersedia pada kolom terakhir `all_ap`; nilai overall resmi tetap diambil dari `box.map`.

### `overall_metrics`

```python
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

- Mengambil precision dan recall rata-rata.
- Menghitung F1 overall.

```python
    return {
        'model': model_label,
        'split': split,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': float(box.map50),
        'mAP50-95': float(box.map),
    }
```

- Mengembalikan dictionary metrik keseluruhan.
- `box.map50`: mAP pada IoU 0.50.
- `box.map`: mAP rata-rata IoU 0.50 sampai 0.95.

### `evaluate_model`

```python
    split_argument = 'val' if split == 'valid' else split
    model = YOLO(str(model_path))
    started = time.perf_counter()
```

- Mengubah nama internal `valid` menjadi nama YAML Ultralytics `val`.
- Memuat weights model.
- Memulai timer evaluasi.

```python
    metrics = model.val(
        data=str(DATA_YAML),
        split=split_argument,
        imgsz=640,
        batch=16,
        conf=0.001,
        plots=True,
        project=str(RUNS_ROOT / 'evaluation'),
        name=run_name,
        exist_ok=True,
        device=DEVICE,
    )
```

- Menjalankan validasi pada split yang dipilih.
- `conf=0.001` menjaga kandidat prediksi untuk PR curve.
- `plots=True` menyimpan confusion matrix, PR curve, F1 curve, dan grafik lain.
- Hasil disimpan di `runs/evaluation/<run_name>`.

```python
    elapsed = time.perf_counter() - started
    save_dir = Path(getattr(metrics, 'save_dir', RUNS_ROOT / 'evaluation' / run_name))
    overall = overall_metrics(metrics, model_label, split)
    per_class = metrics_table(metrics, model_label, split)
```

- Menghitung waktu evaluasi.
- Mengambil folder output evaluasi.
- Menghasilkan metrik overall dan per kelas.

```python
    print(f'{model_label} / {split} selesai dalam {elapsed:.1f} detik')
    print(pd.DataFrame([overall]).to_string(index=False))
    display(per_class.round(4))
    return model, metrics, save_dir, overall, per_class
```

- Mencetak hasil overall.
- Menampilkan tabel per kelas.
- Mengembalikan semua object yang dibutuhkan cell berikutnya.

### `show_evaluation_plots`

```python
    files = ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'PR_curve.png', 'F1_curve.png']
    for filename in files:
        path = save_dir / filename
        if path.exists():
            print(filename)
            display(IPImage(filename=str(path)))
        else:
            print(f'Tidak ditemukan: {path}')
```

- Menentukan nama file grafik Ultralytics.
- Memeriksa apakah file ada.
- Menampilkan file gambar di notebook.
- Memberi pesan jika versi Ultralytics tidak membuat file tertentu.

## Cell 17: Evaluasi baseline

```python
if 'BASELINE_BEST' not in globals() or not Path(BASELINE_BEST).exists():
    print('Baseline best.pt belum ada. Jalankan cell training terlebih dahulu.')
else:
    baseline_valid_model, baseline_valid_metrics, baseline_valid_dir, baseline_valid_overall, baseline_valid_per_class = evaluate_model(
        BASELINE_BEST, 'valid', 'baseline_yolov8s', 'baseline_valid'
    )
    show_evaluation_plots(baseline_valid_dir)
```

- Memastikan training baseline sudah menghasilkan `best.pt`.
- Jika belum, cell tidak crash dan memberi instruksi.
- Jika tersedia, menjalankan evaluasi pada split validation.
- Menyimpan model, object metrics, folder grafik, metrik overall, dan tabel per kelas.
- Menampilkan confusion matrix serta PR/F1 curve.

## Cell 18: Penjelasan eksperimen improvement

### Jenis

Markdown.

### Fungsi

Menjelaskan eksperimen kedua: mengganti `yolov8s` dengan `yolov8m`.

- `yolov8m` biasanya lebih akurat karena model lebih besar.
- Model lebih besar memakai lebih banyak VRAM dan inference lebih lambat.
- Batch diturunkan menjadi 8 untuk mengurangi kebutuhan memori GPU.
- Pemilihan final harus mempertimbangkan mAP dan FPS.

## Cell 19: Training improvement

```python
IMPROVEMENT_MODEL_NAME = 'yolov8m.pt'
IMPROVEMENT_RUN_NAME = 'vehicles_yolov8m_improvement'
```

- Menentukan pretrained model improvement.
- Menentukan nama folder hasil.

```python
improvement_model, improvement_run_dir, IMPROVEMENT_BEST, IMPROVEMENT_INFO = train_model(
    IMPROVEMENT_MODEL_NAME, IMPROVEMENT_RUN_NAME, batch=8, epochs=100
)
```

- Memanggil fungsi training yang sama seperti baseline.
- Hanya ukuran model dan batch yang berubah.
- Menghasilkan weights improvement serta metadata training.

## Cell 20: Bandingkan hasil validasi

```python
valid_results = []
if 'baseline_valid_overall' in globals():
    valid_results.append(baseline_valid_overall)
if 'IMPROVEMENT_BEST' in globals() and Path(IMPROVEMENT_BEST).exists():
    improvement_valid_model, improvement_valid_metrics, improvement_valid_dir, improvement_valid_overall, improvement_valid_per_class = evaluate_model(
        IMPROVEMENT_BEST, 'valid', 'improvement_yolov8m', 'improvement_valid'
    )
    valid_results.append(improvement_valid_overall)
```

- Menyiapkan list hasil validasi.
- Menambahkan hasil baseline jika tersedia.
- Mengevaluasi model improvement pada split valid.
- Menambahkan hasil improvement ke list.

```python
if valid_results:
    comparison_df = pd.DataFrame(valid_results).set_index('model')
    display((comparison_df * 100).round(2).rename(columns={
        'precision': 'precision_%',
        'recall': 'recall_%',
        'f1': 'f1_%',
        'mAP50': 'mAP50_%',
        'mAP50-95': 'mAP50-95_%',
    }))
    print('Pilih model berdasarkan validasi dan FPS, bukan mAP saja.')
else:
    print('Belum ada hasil validasi untuk dibandingkan.')
```

- Mengubah list dictionary menjadi tabel.
- Mengalikan metrik dengan 100 agar tampil sebagai persentase.
- Membulatkan angka ke dua desimal.
- Menegaskan bahwa model terbaik tidak selalu model dengan mAP tertinggi jika FPS terlalu rendah.

## Cell 21: Penjelasan pemilihan model final

### Jenis

Markdown.

### Fungsi

Menjelaskan kapan test dijalankan dan bagaimana `best.pt` dari Colab dapat diupload.

## Cell 22: Upload `best.pt` dari Colab

```python
try:
    from google.colab import files
    uploaded = files.upload()
```

- Mencoba import utilitas upload Colab.
- Membuka dialog upload file.

```python
    if uploaded:
        uploaded_name = next(iter(uploaded))
        shutil.copy2(uploaded_name, WEIGHTS_ROOT / 'best.pt')
        print(f'Disalin ke {WEIGHTS_ROOT / "best.pt"}')
```

- Jika ada file yang diupload, mengambil nama file pertama.
- Menyalin file ke lokasi standar `weights/best.pt`.
- Mencetak lokasi hasil copy.

```python
except ImportError:
    print('Bukan lingkungan Colab. Letakkan best.pt di weights/best.pt secara manual.')
```

- Jika bukan Colab, cell tidak gagal.
- Di luar Colab, file perlu diletakkan manual di `weights/best.pt`.

## Cell 23: Evaluasi final pada test

```python
FINAL_MODEL_PATH = WEIGHTS_ROOT / 'best.pt'
if not FINAL_MODEL_PATH.exists():
    if 'IMPROVEMENT_BEST' in globals() and Path(IMPROVEMENT_BEST).exists():
        FINAL_MODEL_PATH = Path(IMPROVEMENT_BEST)
    elif 'BASELINE_BEST' in globals() and Path(BASELINE_BEST).exists():
        FINAL_MODEL_PATH = Path(BASELINE_BEST)
```

- Default model final adalah `weights/best.pt`.
- Jika belum ada, fallback ke improvement.
- Jika improvement belum ada, fallback ke baseline.

```python
if not FINAL_MODEL_PATH.exists():
    print('Model final belum tersedia. Upload atau set FINAL_MODEL_PATH ke file best.pt.')
else:
    final_model, final_test_metrics, final_test_dir, final_test_overall, final_test_per_class = evaluate_model(
        FINAL_MODEL_PATH, 'test', 'final_model', 'final_test'
    )
    show_evaluation_plots(final_test_dir)
```

- Jika weights belum ada, tampilkan instruksi.
- Jika tersedia, evaluasi sekali pada test split.
- Simpan hasil final dan tampilkan grafik evaluasi.

## Cell 24: Ringkasan pasangan confusion

```python
    confusion = getattr(metrics, 'confusion_matrix', None)
    matrix = np.asarray(getattr(confusion, 'matrix', []), dtype=float)
    if matrix.ndim != 2:
        return pd.DataFrame()
```

- Mengambil object confusion matrix dari hasil Ultralytics.
- Mengubah matrix menjadi NumPy array.
- Jika matrix tidak tersedia atau bentuknya salah, kembalikan DataFrame kosong.

```python
    matrix = matrix[:len(CLASS_NAMES), :len(CLASS_NAMES)]
    rows = []
    for true_id in range(len(CLASS_NAMES)):
        for predicted_id in range(len(CLASS_NAMES)):
            if true_id != predicted_id and matrix[true_id, predicted_id] > 0:
                rows.append({
                    'true_class': CLASS_NAMES[true_id],
                    'predicted_class': CLASS_NAMES[predicted_id],
                    'count': matrix[true_id, predicted_id],
                })
```

- Membatasi matrix ke class ID yang diketahui.
- Mengunjungi setiap kombinasi kelas asli dan kelas prediksi.
- Mengabaikan diagonal karena diagonal adalah prediksi benar.
- Menyimpan pasangan salah klasifikasi yang jumlahnya lebih dari nol.

```python
    return pd.DataFrame(rows).sort_values('count', ascending=False).head(limit) if rows else pd.DataFrame()
```

- Mengurutkan confusion terbesar di atas.
- Membatasi output sebanyak `limit` pasangan.

```python
if 'final_test_metrics' in globals():
    confusion_pairs = top_confusions(final_test_metrics)
    if confusion_pairs.empty:
        print('Tidak ada pasangan confusion off-diagonal yang terbaca.')
    else:
        display(confusion_pairs)
        for row in confusion_pairs.itertuples():
            print(f"{row.true_class} -> {row.predicted_class}: {int(row.count)} kasus; kemungkinan penyebab: bentuk/ukuran visual mirip, occlusion, atau objek terlalu kecil.")
else:
    confusion_pairs = pd.DataFrame()
    print('Evaluasi test belum dijalankan.')
```

- Hanya berjalan jika evaluasi test sudah tersedia.
- Menampilkan tabel pasangan confusion.
- Mencetak interpretasi awal untuk tiap pasangan.
- Penyebab yang dicetak adalah hipotesis; tetap periksa gambar error untuk kesimpulan final.

## Cell 25: Catatan interpretasi confusion dan PR curve

### Jenis

Markdown.

### Fungsi

Mengingatkan agar confusion antara kelas bus tidak langsung dianggap sebagai masalah model tanpa melihat gambar asli.

- Confusion matrix menunjukkan kelas aktual dan prediksi.
- PR curve menunjukkan trade-off precision dan recall pada berbagai threshold.
- Kelas yang bentuknya mirip, tertutup objek lain, atau sangat kecil biasanya lebih sulit dibedakan.

## Cell 26: Helper IoU dan prediction arrays

### `xywhn_to_xyxy`

```python
    class_id, x_center, y_center, box_width, box_height = label
    return np.array([
        (x_center - box_width / 2) * width,
        (y_center - box_height / 2) * height,
        (x_center + box_width / 2) * width,
        (y_center + box_height / 2) * height,
    ], dtype=float)
```

- Mengambil label YOLO normalized.
- Mengubah `center x/y + width/height` menjadi pixel `x1, y1, x2, y2`.
- Class ID tidak masuk ke array koordinat, tetapi dikembalikan secara terpisah oleh pemanggil.

### `box_iou`

```python
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
```

- Menentukan area irisan dua bounding box.

```python
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
```

- Menghitung luas intersection.
- Menghitung luas masing-masing box.
- Menghitung union dengan rumus `area1 + area2 - intersection`.
- Mengembalikan IoU atau nol jika union nol.

### `prediction_arrays`

```python
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4)), np.array([], dtype=int), np.array([], dtype=float)
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidence = result.boxes.conf.cpu().numpy()
    return boxes, classes, confidence
```

- Menangani hasil prediksi tanpa box.
- Mengambil koordinat box, class ID, dan confidence.
- Memindahkan tensor dari GPU ke CPU lalu mengubahnya menjadi NumPy.

## Cell 27: Analisis error

```python
def analyze_error(image_path, model, iou_threshold=0.5):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    ground_truth = [
        (class_id, xywhn_to_xyxy(label, width, height))
        for label in load_yolo_labels(image_path)
        for class_id in [label[0]]
    ]
```

- Membaca ukuran gambar.
- Mengubah setiap label normalized menjadi tuple `(class_id, box)` dalam koordinat pixel.

```python
    result = model.predict(source=str(image_path), imgsz=640, conf=0.001, device=DEVICE, verbose=False)[0]
    predicted_boxes, predicted_classes, predicted_confidence = prediction_arrays(result)
```

- Menjalankan prediksi dengan confidence sangat rendah.
- Mengambil hasil dari gambar pertama.
- Memisahkan koordinat, class ID, dan confidence.

```python
    used_predictions = set()
    matched = []
    false_negative = 0
```

- `used_predictions`: mencegah satu prediksi dipasangkan ke beberapa ground truth.
- `matched`: menyimpan pasangan benar, IoU, dan confidence.
- `false_negative`: menghitung ground truth yang tidak ditemukan.

```python
    for true_class, true_box in ground_truth:
        candidates = [
            (box_iou(true_box, box), index)
            for index, (box, class_id) in enumerate(zip(predicted_boxes, predicted_classes))
            if index not in used_predictions and class_id == true_class
        ]
```

- Untuk setiap ground truth, mencari prediksi dengan class yang sama.
- Prediksi yang sudah dipakai dikecualikan.
- Menghitung IoU setiap kandidat.

```python
        if not candidates or max(candidates)[0] < iou_threshold:
            false_negative += 1
            continue
        iou, prediction_index = max(candidates)
        used_predictions.add(prediction_index)
        matched.append((true_class, iou, predicted_confidence[prediction_index]))
```

- Jika tidak ada kandidat atau IoU di bawah 0.5, object dianggap false negative.
- Kandidat dengan IoU terbesar dipasangkan.
- Prediksi yang sudah dipakai ditandai.
- Pasangan yang berhasil disimpan.

```python
    false_positive = len(predicted_boxes) - len(used_predictions)
    low_confidence = sum(confidence < 0.5 for _, _, confidence in matched)
```

- Prediksi yang tidak terpakai dianggap false positive.
- Match dengan confidence di bawah 0.5 dihitung sebagai low-confidence.

```python
    return {
        'image_path': str(image_path),
        'result': result,
        'ground_truth': ground_truth,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'low_confidence_matches': low_confidence,
        'matched': matched,
        'error_score': 3 * false_positive + 3 * false_negative + low_confidence,
    }
```

- Mengembalikan semua data untuk tabel dan visualisasi.
- `error_score` memprioritaskan FP/FN tiga kali lebih berat daripada low confidence.
- Skor ini adalah ranking heuristic, bukan metrik resmi YOLO.

```python
if 'final_model' in globals() and SPLIT_IMAGES['test']:
    error_items = [analyze_error(path, final_model) for path in SPLIT_IMAGES['test']]
    error_df = pd.DataFrame([
        {key: value for key, value in item.items() if key not in {'result', 'ground_truth', 'matched'}}
        for item in error_items
    ]).sort_values('error_score', ascending=False)
    display(error_df.head(10))
else:
    error_items = []
    error_df = pd.DataFrame()
    print('Model final atau gambar test belum tersedia.')
```

- Menganalisis semua gambar test jika model final tersedia.
- Mengubah ringkasan error menjadi tabel.
- Mengurutkan dari error score terbesar.
- Menampilkan maksimal sepuluh gambar terburuk dalam bentuk tabel.

## Cell 28: Visualisasi error terburuk

```python
    selected = sorted(items, key=lambda item: item['error_score'], reverse=True)[:count]
    if not selected:
        return
```

- Mengambil error terbesar.
- Default menampilkan maksimal delapan contoh.
- Jika tidak ada data, fungsi berhenti.

```python
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 6 * rows))
    axes = np.atleast_1d(axes).ravel()
```

- Membuat layout dua kolom.
- Menyesuaikan jumlah baris dengan jumlah sample.

```python
    for axis, item in zip(axes, selected):
        plotted = item['result'].plot()
        for true_class, true_box in item['ground_truth']:
            x1, y1, x2, y2 = (int(value) for value in true_box)
            cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = CLASS_NAMES[true_class] if 0 <= true_class < len(CLASS_NAMES) else f'class_{true_class}'
            cv2.putText(plotted, f'GT {label}', (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
```

- `result.plot()`: menggambar hasil prediksi model.
- Loop ground truth menambahkan box aktual berwarna hijau.
- Label `GT` membedakan anotasi aktual dari prediksi.
- Dengan begitu, prediksi dan anotasi dapat dibandingkan secara visual.

```python
        axis.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{Path(item['image_path']).name} | FP={item['false_positive']} FN={item['false_negative']} low-conf={item['low_confidence_matches']}")
        axis.axis('off')
```

- Mengubah BGR ke RGB sebelum ditampilkan.
- Judul mencantumkan nama file, jumlah FP, FN, dan low confidence.

```python
    for axis in axes[len(selected):]:
        axis.axis('off')
    figure.suptitle('Contoh prediksi dengan error terbesar', fontsize=16)
    figure.tight_layout()
    plt.show()

display_error_examples(error_items, count=8)
```

- Mematikan subplot kosong.
- Merapikan dan menampilkan figure.
- Memanggil visualisasi untuk delapan error terbesar.

## Cell 29: Recall berdasarkan ukuran objek

```python
    rows = []
    for item in items:
        image = cv2.imread(item['image_path'])
        height, width = image.shape[:2]
        predicted_boxes, predicted_classes, _ = prediction_arrays(item['result'])
```

- Menyiapkan baris analisis ukuran.
- Membaca ukuran gambar.
- Mengambil box dan class prediksi.

```python
        for label in load_yolo_labels(item['image_path']):
            true_class = label[0]
            true_box = xywhn_to_xyxy(label, width, height)
            area_ratio = label[3] * label[4]
            bucket = size_bucket(area_ratio)
            matched = any(class_id == true_class and box_iou(true_box, box) >= 0.5 for box, class_id in zip(predicted_boxes, predicted_classes))
            rows.append({'size': bucket, 'matched': matched})
```

- Mengubah setiap label ke pixel box.
- Menghitung area ratio dan kategori small/medium/large.
- Menganggap object matched jika class sama dan IoU minimal 0.5.
- Menyimpan status match untuk agregasi.

```python
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).groupby('size').agg(objects=('matched', 'size'), matched=('matched', 'sum'))
    result['recall_at_iou50'] = result['matched'] / result['objects']
    return result.sort_index()
```

- Jika tidak ada data, kembalikan tabel kosong.
- Menghitung jumlah object dan jumlah match per kategori ukuran.
- Menghitung recall approximate pada IoU 0.5.

```python
if error_items:
    print('Recall berdasarkan ukuran objek (prediksi conf=0.001, IoU=0.5):')
    display(size_recall(error_items).round(4))
```

- Menampilkan hasil recall per ukuran jika error analysis sudah dijalankan.
- Angka ini diagnostik sederhana, bukan AP resmi per ukuran.

## Cell 30: Judul inference

### Jenis

Markdown.

### Fungsi

Menjelaskan bahwa inference gambar/video dipakai untuk memeriksa data unseen dan mengukur FPS pada hardware aktual.

## Cell 31: Inference pada gambar test

```python
if 'final_model' in globals() and SPLIT_IMAGES['test']:
    sample_paths = random.Random(SEED).sample(SPLIT_IMAGES['test'], min(5, len(SPLIT_IMAGES['test'])))
```

- Memastikan model final dan test image tersedia.
- Memilih maksimal lima gambar test secara reproducible.

```python
    prediction_dir = RUNS_ROOT / 'inference' / 'test_images'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(sample_paths), 1, figsize=(14, 5 * len(sample_paths)))
    axes = np.atleast_1d(axes).ravel()
```

- Menentukan folder output gambar hasil inference.
- Membuat folder jika belum ada.
- Menyiapkan satu subplot per gambar.

```python
    for axis, image_path in zip(axes, sample_paths):
        result = final_model.predict(source=str(image_path), imgsz=640, conf=0.25, device=DEVICE, verbose=False)[0]
        plotted = result.plot()
        cv2.imwrite(str(prediction_dir / image_path.name), plotted)
        axis.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        axis.set_title(image_path.name)
        axis.axis('off')
```

- Menjalankan inference dengan threshold confidence 0.25.
- Menggambar box hasil prediksi.
- Menyimpan gambar hasil ke folder inference.
- Menampilkan hasil pada notebook.

```python
    figure.tight_layout()
    plt.show()
else:
    print('Model final atau test split belum tersedia.')
```

- Merapikan dan menampilkan figure.
- Jika prasyarat belum ada, tampilkan pesan.

## Cell 32: Inference pada video

```python
VIDEO_PATH = PROJECT_ROOT / 'input_video.mp4'
VIDEO_OUTPUT = RUNS_ROOT / 'inference' / 'vehicles_detected.mp4'
```

- `VIDEO_PATH`: video input yang harus diletakkan di root project.
- `VIDEO_OUTPUT`: lokasi video dengan bounding box.

```python
# from google.colab import files
# uploaded_video = files.upload()
# VIDEO_PATH = Path(next(iter(uploaded_video)))
```

- Contoh opsional untuk upload video di Colab.
- Baris masih dikomentari agar tidak otomatis membuka dialog upload.

### `run_video_inference`

```python
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Video tidak dapat dibuka: {video_path}')
```

- Membuka video melalui OpenCV.
- Jika gagal dibuka, hentikan dengan error yang jelas.

```python
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), source_fps or 30.0, (width, height))
```

- Mengambil ukuran frame.
- Mengambil FPS asli video.
- Membuat writer untuk video output.
- Jika FPS source tidak tersedia, memakai 30 FPS sebagai fallback output.

```python
    timings = []
    frame_count = 0
    preview = None
    while True:
        success, frame = capture.read()
        if not success:
            break
```

- `timings`: menyimpan durasi inference setiap frame.
- `frame_count`: menghitung jumlah frame.
- `preview`: menyimpan frame hasil pertama.
- Loop membaca frame hingga video selesai.

```python
        started = time.perf_counter()
        result = model.predict(source=frame, imgsz=640, conf=0.25, device=DEVICE, verbose=False)[0]
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        annotated = result.plot()
        writer.write(annotated)
```

- Memulai timer.
- Menjalankan YOLO pada frame.
- Menghitung durasi inference.
- Menyimpan durasi.
- Menggambar hasil prediksi.
- Menulis frame hasil ke video output.

```python
        if preview is None:
            preview = annotated.copy()
        frame_count += 1
```

- Menyimpan frame pertama sebagai preview.
- Menambah counter frame.

```python
    capture.release()
    writer.release()
    if not timings:
        raise ValueError('Video tidak memiliki frame.')
    warmup = timings[1:] if len(timings) > 1 else timings
    mean_seconds = float(np.mean(warmup))
```

- Menutup input dan output video.
- Menolak video kosong.
- Mengabaikan frame pertama dari perhitungan jika memungkinkan karena GPU biasanya memiliki warm-up.
- Menghitung rata-rata durasi inference.

```python
    return {
        'frames': frame_count,
        'source_fps': source_fps,
        'inference_fps_mean': 1 / mean_seconds if mean_seconds else float('inf'),
        'inference_ms_mean': mean_seconds * 1000,
        'inference_fps_median': 1 / float(np.median(warmup)) if np.median(warmup) else float('inf'),
        'preview': preview,
    }
```

- Mengembalikan jumlah frame dan FPS source.
- `inference_fps_mean`: FPS berdasarkan rata-rata durasi inference.
- `inference_ms_mean`: latency rata-rata dalam milidetik.
- `inference_fps_median`: FPS berdasarkan median latency.
- `preview`: frame hasil untuk ditampilkan.

```python
if 'final_model' in globals() and VIDEO_PATH.exists():
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    video_stats = run_video_inference(VIDEO_PATH, VIDEO_OUTPUT, final_model)
    print({key: value for key, value in video_stats.items() if key != 'preview'})
```

- Menjalankan inference hanya jika model dan video tersedia.
- Membuat folder output.
- Menampilkan statistik tanpa object image preview.

```python
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(video_stats['preview'], cv2.COLOR_BGR2RGB))
    plt.title(f"Contoh frame | {video_stats['inference_fps_mean']:.2f} FPS inference")
    plt.axis('off')
    plt.show()
    print(f'Video output: {VIDEO_OUTPUT}')
else:
    print(f'Letakkan video pendek di {VIDEO_PATH} lalu jalankan ulang cell ini.')
```

- Menampilkan preview frame.
- Menulis FPS pada judul.
- Menyembunyikan axis.
- Mencetak path video output.
- Jika video belum ada, menampilkan lokasi yang diharapkan.

## Cell 33: Penjelasan generate report

### Jenis

Markdown.

### Fungsi

Menjelaskan bahwa cell berikut akan membuat report Markdown berdasarkan hasil yang tersedia dari notebook.

## Cell 34: Generate report Markdown

### `metric_line`

```python
    if not metrics_dict:
        return 'Belum tersedia'
    return (f"precision={metrics_dict['precision']:.4f}, recall={metrics_dict['recall']:.4f}, "
            f"F1={metrics_dict['f1']:.4f}, mAP50={metrics_dict['mAP50']:.4f}, "
            f"mAP50-95={metrics_dict['mAP50-95']:.4f}")
```

- Mengubah dictionary metrik menjadi satu baris ringkas.
- Jika metrik belum ada, menulis `Belum tersedia`.
- `.4f` menampilkan empat angka di belakang koma.

### Dataset dan hardware report

```python
report_lines = [
    '# Vehicles YOLO Report',
    '',
    '## Dataset',
    f'- Location: `{DATASET_ROOT}`',
    '- Classes: ' + ', '.join(CLASS_NAMES),
    f'- Images: ' + ', '.join(f'{split}={len(paths)}' for split, paths in SPLIT_IMAGES.items()),
    '',
    '## Hardware dan konfigurasi',
    f'- Device: `{DEVICE}`',
    f'- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}`',
    '- Image size: 640, baseline epochs: 100, patience: 20',
    '',
    '## Validasi model',
]
```

- Membuat list baris report.
- Menulis lokasi dataset, nama kelas, jumlah gambar, device, GPU, dan konfigurasi training.

### Hasil validasi

```python
if valid_results:
    for result in valid_results:
        report_lines.append(f"- {result['model']}: {metric_line(result)}")
else:
    report_lines.append('- Belum ada hasil validasi.')
```

- Menambahkan metrik setiap model yang sudah dievaluasi.
- Jika belum ada hasil, report tetap dibuat dengan status belum tersedia.

### Hasil test dan analisis

```python
report_lines.extend(['', '## Test final', f'- Model: `{FINAL_MODEL_PATH}`'])
report_lines.append(f"- {metric_line(globals().get('final_test_overall'))}")
report_lines.extend([
    '',
    '## Analisis',
    '- Confusion matrix dan PR curve tersimpan di folder `runs/evaluation/`.',
    '- Error analysis menampilkan sampel FP, FN, dan matched prediction dengan confidence rendah.',
    '- Recall per ukuran objek dihitung secara approximate pada IoU 0.5; angka ini bukan pengganti AP resmi per ukuran.',
])
```

- Menambahkan path model final.
- Menambahkan metrik test final jika tersedia.
- Menambahkan lokasi artefak evaluasi dan penjelasan batasan recall ukuran objek.

### Hasil video dan kesimpulan

```python
if 'video_stats' in globals():
    report_lines.extend([
        '',
        '## Inference video',
        f"- Mean inference FPS: {video_stats['inference_fps_mean']:.2f}",
        f"- Median inference FPS: {video_stats['inference_fps_median']:.2f}",
        f"- Target 10 FPS tercapai: {video_stats['inference_fps_mean'] >= 10}",
    ])
else:
    report_lines.extend(['', '## Inference video', '- Belum dijalankan; tambahkan video dan jalankan cell inference.'])
```

- Menambahkan FPS mean dan median jika video sudah diuji.
- Menandai apakah target 10 FPS tercapai.
- Jika belum ada video, report tetap menyebutkan langkah yang belum dilakukan.

```python
report_lines.extend([
    '',
    '## Kesimpulan',
    '- Gunakan model final untuk tahap OpenCV + WebSocket setelah metrik test dan FPS sesuai kebutuhan.',
    '- Jika kelas bus masih sering tertukar, periksa kualitas label, tambah data kelas tersebut, dan evaluasi threshold confidence/NMS pada video target.',
])
REPORT_PATH = REPORTS_ROOT / 'report.md'
REPORT_PATH.write_text('\n'.join(report_lines) + '\n')
print(REPORT_PATH.read_text())
```

- Menambahkan kesimpulan dan saran next step.
- Menentukan path `reports/report.md`.
- Menggabungkan semua baris dengan newline.
- Menulis report ke disk.
- Membaca dan mencetak report agar langsung terlihat di notebook.

## Struktur output yang diharapkan

Setelah notebook selesai dijalankan, struktur utama biasanya seperti ini:

```text
datasets/vehicles/
├── data.yaml
├── train/
├── valid/
└── test/

runs/
├── vehicles_yolov8s_baseline/
├── vehicles_yolov8m_improvement/
├── evaluation/
└── inference/

weights/
└── best.pt

reports/
└── report.md
```

## Catatan penting

- Jangan memakai test split untuk memilih model atau mengubah hyperparameter.
- Jika API key pernah dipublikasikan, revoke lalu buat key baru.
- `mAP50-95` lebih ketat daripada `mAP50` karena mengevaluasi beberapa threshold IoU.
- Confusion matrix menunjukkan pola kesalahan klasifikasi, bukan hanya jumlah object benar.
- FPS pada Colab belum tentu sama dengan FPS pada komputer target pipeline realtime.
- Jika cell upload dipakai di Colab, `best.pt` hanya ada pada session tersebut sampai diunduh atau disalin ke storage permanen.
