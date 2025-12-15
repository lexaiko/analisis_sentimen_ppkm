# Klasifikasi Sentimen Tweet PPKM

Aplikasi web untuk mengklasifikasikan sentimen tweet tentang PPKM (Pemberlakuan Pembatasan Kegiatan Masyarakat) menggunakan Machine Learning dan Streamlit.

## 📊 Dataset

Dataset yang digunakan: [Twitter Dataset PPKM](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm/data?select=INA_TweetsPPKM_Labeled_Pure.csv)

Dataset berisi tweet dalam bahasa Indonesia dengan 3 label sentimen:
- **0** = Positif 😊
- **1** = Netral 😐
- **2** = Negatif 😞

## 🚀 Instalasi

### 1. Clone atau Download Repository

```bash
cd c:\semester5\spk\analisis_sentimen
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📖 Cara Penggunaan

### Step 1: Training Model

Jalankan script untuk melatih model machine learning:

```bash
python train_model.py
```

Script ini akan:
- Load dan preprocess dataset
- Melatih model Logistic Regression
- Evaluasi performa model
- Menyimpan model ke folder `models/`
- Menampilkan confusion matrix dan classification report

**Output:**
- `models/sentiment_model.pkl` - Model terlatih
- `models/vectorizer.pkl` - TF-IDF Vectorizer
- `models/confusion_matrix.png` - Visualisasi confusion matrix

### Step 2: Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 🎯 Fitur Aplikasi

### 1. Klasifikasi Single Comment
- Input komentar/tweet secara manual
- Lihat hasil klasifikasi (Positif/Netral/Negatif)
- Lihat distribusi probabilitas
- Lihat hasil preprocessing teks

### 2. Batch Processing
- Upload file CSV dengan multiple comments
- Proses semua comment sekaligus
- Download hasil dalam format CSV
- Lihat statistik sentimen

## 📁 Struktur Project

```
analisis_sentimen/
│
├── data/
│   └── INA_TweetsPPKM_Labeled_Pure.csv  # Dataset
│
├── models/
│   ├── sentiment_model.pkl              # Model terlatih
│   ├── vectorizer.pkl                   # TF-IDF Vectorizer
│   └── confusion_matrix.png             # Confusion matrix
│
├── preprocessing.py                     # Module preprocessing teks
├── train_model.py                       # Script training model
├── app.py                              # Aplikasi Streamlit
├── requirements.txt                     # Dependencies
└── README.md                           # Dokumentasi
```

## 🛠️ Teknologi yang Digunakan

### Machine Learning
- **Scikit-learn** - Framework ML
- **Logistic Regression** - Algoritma klasifikasi
- **TF-IDF** - Feature extraction

### Text Preprocessing
- **Sastrawi** - Stemming dan stopword removal untuk Bahasa Indonesia
- **Regex** - Text cleaning

### Web Application
- **Streamlit** - Framework web app
- **Matplotlib & Seaborn** - Visualisasi data

## 📊 Performa Model

Model Logistic Regression dengan TF-IDF vectorization memberikan hasil yang baik untuk klasifikasi sentimen tweet PPKM.

Untuk melihat detail performa:
1. Jalankan `train_model.py`
2. Lihat classification report di terminal
3. Lihat confusion matrix di `models/confusion_matrix.png`

## 💡 Contoh Penggunaan

### Contoh Input Positif:
```
"PPKM sangat efektif dalam menurunkan angka kasus COVID-19"
```

### Contoh Input Netral:
```
"Pemerintah mengumumkan perpanjangan PPKM level 2 hingga akhir bulan"
```

### Contoh Input Negatif:
```
"PPKM membuat ekonomi rakyat semakin terpuruk dan banyak yang kehilangan pekerjaan"
```

## 🔧 Troubleshooting

### Model belum tersedia
Jika muncul pesan error "Model belum tersedia":
```bash
python train_model.py
```

### Error import Sastrawi
Jika ada error saat import Sastrawi:
```bash
pip install --upgrade Sastrawi
```

### Port sudah digunakan
Jika port 8501 sudah digunakan:
```bash
streamlit run app.py --server.port 8502
```

## 📝 Catatan

- Dataset harus ada di folder `data/INA_TweetsPPKM_Labeled_Pure.csv`
- Training model membutuhkan waktu beberapa menit tergantung spesifikasi komputer
- Untuk batch processing, file CSV harus memiliki kolom `text` atau `Tweet`

## 👨‍💻 Developer

Dibuat untuk tugas SPK (Sistem Pendukung Keputusan) Semester 5

## 📄 Lisensi

Dataset source: [Kaggle - Twitter Dataset PPKM](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm/)
