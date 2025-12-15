# 📚 RANGKUMAN SISTEM PENDUKUNG KEPUTUSAN - ANALISIS SENTIMEN TWEET PPKM

## 1️⃣ OVERVIEW PROYEK

### Judul
**Aplikasi Klasifikasi Sentimen Tweet PPKM (Pemberlakuan Pembatasan Kegiatan Masyarakat)**

### Tujuan
- Mengklasifikasikan sentimen tweet dalam bahasa Indonesia tentang PPKM
- Menentukan apakah tweet bersifat Positif, Netral, atau Negatif
- Membantu pemerintah/organisasi memahami opini publik terhadap kebijakan PPKM

### Jenis Masalah
**Multi-Class Text Classification** dengan 3 kelas sentimen:
- **0 = Positif** (mendukung/pujian terhadap PPKM)
- **1 = Netral** (informatif, tanpa opini jelas)
- **2 = Negatif** (kritik/keluhan terhadap PPKM)

---

## 2️⃣ DATASET

### Sumber Data
- **Nama Dataset**: Twitter Dataset PPKM (Indonesia)
- **File**: `INA_TweetsPPKM_Labeled_Pure.csv`
- **Format**: Tab-separated values (TSV)
- **Bahasa**: Bahasa Indonesia

### Karakteristik Data
- **Total Samples**: 8,067 tweets (sebelum preprocessing)
- **Fitur Utama**: 
  - `Tweet`: Teks tweet/komentar
  - `sentiment`: Label kelas (0, 1, 2)
- **Distribusi Class**: Terseimbang (balanced dataset)
- **Pre-labeled**: Data sudah berlabel sentiment

### Data Exploration
```
Label 0 (Positif): ~2,700 samples (33%)
Label 1 (Netral):  ~2,700 samples (33%)
Label 2 (Negatif): ~2,667 samples (34%)
```

---

## 3️⃣ METODOLOGI & TAHAPAN

### A. PREPROCESSING (Text Cleaning)

**Tujuan**: Membersihkan data teks agar lebih konsisten dan siap untuk modeling

**Langkah-Langkah Preprocessing** (dalam file `preprocessing.py`):

1. **Lowercase**
   - Mengubah semua huruf menjadi huruf kecil
   - Alasan: Menghindari duplikasi feature ("PPKM" = "ppkm")

2. **Remove URL**
   - Regex: `http\S+|www\S+|https\S+`
   - Alasan: URL tidak mengandung informasi sentimen

3. **Remove Mention (@username)**
   - Regex: `@\w+`
   - Alasan: Nama pengguna tidak relevan untuk sentimen

4. **Remove Hashtag (#)**
   - Regex: `#\w+`
   - Alasan: Hashtag tag sudah diwakili oleh kata-kata
   - Catatan: Bisa dipertimbangkan untuk dipertahankan di beberapa kasus

5. **Remove Angka**
   - Regex: `\d+`
   - Alasan: Angka kurang berkontribusi pada sentimen

6. **Remove Punctuation & Special Characters**
   - Hapus tanda baca dan karakter khusus
   - Alasan: Noise untuk model ML

7. **Remove Extra Whitespace**
   - Normalize whitespace ke single space
   - Alasan: Konsistensi format

8. **Remove Stopwords**
   - Library: **Sastrawi** (Indonesian Stopword Remover)
   - Stopwords: "yang", "dan", "atau", "untuk", dll
   - Alasan: Kata-kata umum tidak membedakan sentimen

9. **Stemming**
   - Library: **Sastrawi Stemmer Factory**
   - Contoh: "berkembang" → "kembang", "membantu" → "bantu"
   - Alasan: Mengurangi dimensionality, menganggap kata dasar sama

**Contoh Preprocessing**:
```
Original: "@user PPKM bagus banget!! 😊 https://t.co/xxx #ppkm2021"
Cleaned:  "bagus banget"
```

### B. FEATURE EXTRACTION (TF-IDF Vectorization)

**Teknik**: TF-IDF (Term Frequency - Inverse Document Frequency)

**Konfigurasi**:
```python
TfidfVectorizer(
    max_features=5000,      # Ambil 5000 fitur teratas
    ngram_range=(1, 2)      # Unigram & Bigram
)
```

**Penjelasan**:
- **TF (Term Frequency)**: Berapa banyak kata muncul di dokumen
- **IDF (Inverse Document Frequency)**: Seberapa unik/penting kata di corpus
- **TF-IDF Value** = TF × IDF (bobot kata)

**Mengapa TF-IDF?**
- Mengurangi pengaruh kata-kata umum
- Memberikan bobot tinggi pada kata yang paling diskriminatif
- Lebih baik dari simple bag-of-words
- Efisien secara komputasi

**Output**: Sparse matrix (8,067 dokumen × 5,000 fitur)

### C. MODEL MACHINE LEARNING

**Model Utama**: **Logistic Regression**

**Alasan Pemilihan Logistic Regression**:
1. ✅ Cocok untuk multi-class classification (3 kelas)
2. ✅ Sederhana tapi efektif untuk text classification
3. ✅ Interpretable (bisa lihat feature importance)
4. ✅ Cepat training
5. ✅ Memberikan probability output
6. ✅ Scalable untuk dataset besar

**Konfigurasi**:
```python
LogisticRegression(
    max_iter=1000,      # Max iterasi untuk convergence
    random_state=42     # Reproducibility
)
```

**Algoritma Lainnya yang Dibandingkan** (dalam `compare_models()` function):
- **Naive Bayes** (MultinomialNB): Probabilistic, cocok untuk text
- **SVM** (Support Vector Machine): Powerful tapi slower
- **Random Forest**: Ensemble method, handling non-linearity

**Performa Umum**:
- Logistic Regression: ~80-85% accuracy
- Naive Bayes: ~75-80% accuracy
- Random Forest: ~78-83% accuracy

### D. SPLIT DATA

**Strategi**: Train-Test Split dengan Stratification

```python
train_test_split(
    X, y, 
    test_size=0.2,          # 80% train, 20% test
    random_state=42,        # Reproducible
    stratify=y              # Maintain class distribution
)
```

**Hasil**:
- Training set: ~6,453 samples (80%)
- Test set: ~1,614 samples (20%)

**Mengapa Stratification?**
- Memastikan distribusi class sama di train & test
- Penting untuk imbalanced dataset (tidak di sini, tapi best practice)

---

## 4️⃣ EVALUASI MODEL

### Metrics yang Digunakan

#### 1. **Accuracy**
```
Accuracy = (TP + TN) / Total
```
- **Definisi**: Persentase prediksi yang benar
- **Range**: 0-1 (0-100%)
- **Penggunaan**: Overall performance metric
- **Kekurangan**: Bisa misleading untuk imbalanced data

#### 2. **Precision** (per kelas)
```
Precision = TP / (TP + FP)
```
- **Definisi**: Dari prediksi POSITIF, berapa yang benar?
- **Artinya**: Ketepatan prediksi class
- **Contoh**: Precision untuk "Negatif" = 95% artinya ketika model bilang NEGATIF, 95% benar

#### 3. **Recall** (per kelas)
```
Recall = TP / (TP + FN)
```
- **Definisi**: Dari data actual POSITIF, berapa yang terdeteksi?
- **Artinya**: Sensitivity/coverage untuk class
- **Contoh**: Recall untuk "Negatif" = 88% artinya 88% tweet negatif terdeteksi

#### 4. **F1-Score** (per kelas)
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Definisi**: Harmonic mean antara Precision & Recall
- **Kegunaan**: Balance antara precision dan recall
- **Preferred**: Ketika tradeoff precision-recall penting

#### 5. **Confusion Matrix**
```
                Predicted
              Pos  Net  Neg
Actual Pos    TP   FP   FP
       Net    FN   TP   FP
       Neg    FN   FN   TP
```
- **Guna**: Visualisasi error patterns
- **Analisis**: Lihat misclassifications terbanyak
- **Output**: Heatmap dengan seaborn

### Interpretasi Classification Report
```
              precision    recall  f1-score   support
   Positif       0.82      0.80      0.81       540
   Netral        0.81      0.83      0.82       540
   Negatif       0.85      0.85      0.85       534
 
   accuracy                           0.83      1614
   macro avg      0.83      0.83      0.83      1614
weighted avg      0.83      0.83      0.83      1614
```

**Penjelasan**:
- **precision**: Ketepatan per class
- **recall**: Coverage per class
- **f1-score**: Balance metric
- **support**: Jumlah sample per class di test set
- **macro avg**: Average tanpa weight
- **weighted avg**: Average dengan weight berdasarkan support

---

## 5️⃣ SISTEM APLIKASI

### Frontend: Streamlit Web Application

**File**: `app.py`

**Fitur Utama**:

#### 1. **Single Sentiment Analysis**
- Input: Teks komentar/tweet
- Output: 
  - Klasifikasi sentimen (0/1/2)
  - Confidence score (%)
  - Distribusi probabilitas per kelas
  - Detail preprocessing

#### 2. **Batch Processing**
- Input: Upload CSV file
- Output: 
  - CSV hasil dengan prediksi
  - Statistik agregat
  - Download hasil

#### 3. **Visualisasi**
- **Bar Chart Probabilitas**: Horizontal bar chart
- **Confusion Matrix**: Heatmap (saat training)
- **Statistik Batch**: Metrics cards

**Technology Stack**:
- **Framework**: Streamlit (Python web framework)
- **Deployment**: Localhost (default: http://localhost:8501)
- **Styling**: Custom CSS untuk sentiment boxes

---

## 6️⃣ TEKNOLOGI & LIBRARY

```
pandas==2.0.3              # Data manipulation
numpy==1.24.3              # Numerical computing
scikit-learn==1.3.0        # Machine learning
  - LogisticRegression
  - TfidfVectorizer
  - train_test_split
  - classification_report, confusion_matrix
streamlit==1.28.0          # Web UI
matplotlib==3.7.2          # Plotting
seaborn==0.12.2            # Statistical visualization
Sastrawi==1.0.1            # Indonesian NLP
  - Stemmer
  - StopWordRemover
```

---

## 7️⃣ ALUR SISTEM

```
┌─────────────────┐
│  Raw Dataset    │ (CSV tweets)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │ (clean, remove stopwords, stem)
│  (TextProcess)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │ (TF-IDF Vectorization)
│   (5000 dims)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Split   │ (80-20 train-test)
    └────┬────┘
         │
    ┌────┴────────────┐
    │ Train Set       │ Test Set
    │ (6453 samples)  │ (1614 samples)
    └────┬────────────┘
         │
         ▼
┌────────────────────────┐
│ Train Model            │
│ LogisticRegression     │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Save Model & Vectorizer│ (pickle files)
│ To models/ folder      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Evaluate on Test Set   │ 
│ (Accuracy, Precision,  │
│  Recall, F1, CM)       │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────────┐
│  Deploy in Streamlit App   │
│  Load model & vectorizer   │
│  Predict sentimen baru     │
└────────────────────────────┘
```

---

## 8️⃣ KEMUNGKINAN PERTANYAAN UJIAN

### A. KONSEP UMUM

**Q1: Jelaskan apa itu Sistem Pendukung Keputusan (SPK)?**
- Definisi: Sistem berbasis komputer interaktif untuk membantu pengambilan keputusan
- Komponen: Data, Model, User interface
- Tujuan: Meningkatkan kualitas keputusan
- Contoh aplikasi Anda: Membantu memahami sentimen publik untuk kebijakan

**Q2: Kategori SPK apa yang digunakan proyek ini?**
- **Tipe**: DSS berbasis Model (model-driven)
- Fokus: Machine Learning untuk klasifikasi
- Bukan OLAP/data warehouse focused

**Q3: Jelaskan tahapan dalam data mining**
- Data Collection → Data Preprocessing → Feature Engineering → Model Building → Evaluation → Deployment

**Q4: Apa perbedaan supervised vs unsupervised learning?**
- **Supervised**: Punya label (Anda: classification, 3 label sentimen)
- **Unsupervised**: Tanpa label (clustering, association rule)
- Proyek Anda adalah **Supervised Learning**

### B. DATASET & PREPROCESSING

**Q5: Berapa jumlah data dan distribusinya?**
- Total: 8,067 tweets
- Distribusi: Balanced (±33% per kelas)
- Train: 6,453 (80%)
- Test: 1,614 (20%)

**Q6: Sebutkan dan jelaskan tahapan preprocessing!**
1. **Lowercase**: Normalisasi huruf
2. **URL removal**: Hapus link
3. **Mention removal**: Hapus @username
4. **Hashtag removal**: Hapus #
5. **Number removal**: Hapus angka
6. **Punctuation removal**: Hapus tanda baca
7. **Whitespace normalization**: Hapus space berlebih
8. **Stopword removal**: Hapus kata umum (Sastrawi)
9. **Stemming**: Reduksi ke kata dasar (Sastrawi)

**Q7: Apa itu TF-IDF? Mengapa digunakan?**
- **TF-IDF**: Term Frequency - Inverse Document Frequency
- **Cara kerja**: TF = berapa sering kata / IDF = keunikan kata di corpus
- **Keuntungan**: 
  - Bobot tinggi untuk kata diskriminatif
  - Bobot rendah untuk kata umum
  - Dimensionality reduction

**Q8: Bagaimana Sastrawi membantu dalam preprocessing?**
- **Stemmer**: Menghilangkan afiks (prefix/suffix)
  - "berkembang" → "kembang"
  - "membantu" → "bantu"
- **Stopword Remover**: Menghapus kata-kata umum bahasa Indonesia
  - Menurunkan noise, fokus pada kata penting
- Spesifik untuk bahasa Indonesia

**Q9: Mengapa menggunakan stratified train-test split?**
- Memastikan distribusi class sama di train dan test
- Penting untuk imbalanced dataset
- Mencegah bias evaluasi

### C. MODEL MACHINE LEARNING

**Q10: Model ML apa yang digunakan? Mengapa?**
- **Model**: Logistic Regression
- **Alasan**:
  1. Cocok untuk multi-class classification
  2. Sederhana & interpretable
  3. Cepat training
  4. Output probability
  5. Scalable

**Q11: Jelaskan cara kerja Logistic Regression**
- **Input**: TF-IDF feature vector (5000 dimensi)
- **Proses**: Hitung probabilitas untuk setiap class menggunakan sigmoid function
- **Output**: Probability distribution [P(Positif), P(Netral), P(Negatif)]
- **Decision**: Class dengan probability tertinggi dipilih

**Q12: Perbandingan dengan model lain?**
- **Naive Bayes**: Lebih cepat, tapi asumsi independence terlalu kuat
- **SVM**: Lebih akurat tapi lebih lambat training
- **Random Forest**: Good performance tapi less interpretable
- **Logistic Regression**: Balance antara accuracy & efficiency ✓

**Q13: Apa parameter penting dalam Logistic Regression?**
- `max_iter=1000`: Jumlah iterasi untuk convergence
- `random_state=42`: Reproducibility
- `multi_class`: Handling 3 kelas (default: multinomial)

**Q14: Apa itu max_features di TfidfVectorizer?**
- Membatasi jumlah features ke 5000 teratas
- Alasan: Mengurangi dimensionality, noise reduction
- Trade-off: Informasi mungkin hilang

**Q15: Apa itu ngram_range=(1,2)?**
- `(1,2)`: Unigram (single word) + Bigram (2 consecutive words)
- Contoh Unigram: "PPKM", "bagus"
- Contoh Bigram: "PPKM bagus", "ekonomi hancur"
- Mengapa: Capture local context/meaning

### D. EVALUASI MODEL

**Q16: Jelaskan Confusion Matrix**
- 3×3 matrix (3 kelas)
- Diagonal = correct predictions
- Off-diagonal = errors
- Identifikasi: Kelas mana yang sering salah klasifikasi?

**Q17: Apa itu Accuracy? Kekurangannya?**
- **Accuracy** = (TP + TN) / Total
- **Kekurangan**: Bisa misleading untuk imbalanced data
- **Contoh**: Dataset dengan 95% kelas A, model predict semua A → 95% accuracy tapi useless

**Q18: Jelaskan Precision vs Recall**
- **Precision**: "Ketepatan" - dari prediksi positif, berapa yang benar?
  - Important when: False positive costly (e.g., medical diagnosis)
- **Recall**: "Coverage" - dari data actual positif, berapa terdeteksi?
  - Important when: Missing positive costly (e.g., disease detection)
- **Trade-off**: Increase satu, decrease yang lain

**Q19: Kapan menggunakan F1-Score?**
- Ketika perlu balance antara Precision & Recall
- Harmonic mean: tidak biased ke nilai extreme
- Metric single-number untuk evaluasi overall

**Q20: Interpretasi F1-Score = 0.83**
- Baik (range 0-1, 1 sempurna)
- Lebih dari 80% adalah acceptable untuk text classification
- Trade-off precision-recall well balanced

### E. APLIKASI & DEPLOYMENT

**Q21: Teknologi apa yang digunakan untuk web app?**
- **Streamlit**: Python framework untuk rapid prototyping
- **Keuntungan**: Cepat buat, minimal coding, integration dengan ML
- **Kekurangan**: Less flexible untuk complex UI

**Q22: Fitur apa saja di aplikasi?**
1. Single text classification dengan confidence
2. Batch processing CSV
3. Visualisasi probabilitas (bar chart)
4. Detail preprocessing view
5. Download hasil CSV

**Q23: Bagaimana proses prediksi di aplikasi?**
1. User input teks
2. Preprocess dengan TextPreprocessor
3. TF-IDF transform dengan vectorizer
4. Logistic Regression predict
5. Get probability & label
6. Display hasil + visualisasi

**Q24: Apa format output prediksi?**
- **Label**: 0 (Positif), 1 (Netral), 2 (Negatif)
- **Probability**: Array 3 nilai (sum = 1.0)
  - Contoh: [0.85, 0.10, 0.05] → Positif dengan confidence 85%

**Q25: Bagaimana batch processing bekerja?**
- Load CSV file
- Loop setiap baris → preprocess → predict
- Collect semua predictions
- Add kolom baru 'prediction' & 'sentiment_label'
- Display tabel + statistik
- Download as CSV

### F. PROBLEM SOLVING & TROUBLESHOOTING

**Q26: Apa masalah jika accuracy training tinggi tapi test rendah?**
- **Overfitting**: Model memorize training data
- **Solusi**: 
  - Regularization (L1/L2)
  - Reduce model complexity
  - More data
  - Feature selection

**Q27: Bagaimana jika class imbalanced?**
- **Problem**: Model bias ke majority class
- **Solusi**:
  - Stratified split (sudah digunakan)
  - Weighted loss function
  - SMOTE/undersampling
  - Adjust threshold

**Q28: Error "Model not found" di aplikasi**
- **Penyebab**: Model belum di-train (run `train_model.py` dulu)
- **Solusi**: 
  ```bash
  python train_model.py
  ```

**Q29: Accuracy rendah, apa yang bisa dilakukan?**
1. **EDA**: Analisis dataset lebih dalam
2. **Feature Engineering**: Tambah/ganti features
3. **Preprocessing**: Eksperimen parameter (stemming on/off, min df, dll)
4. **Model**: Try model lain
5. **Hyperparameter**: Tuning (GridSearchCV)
6. **Data**: Lebih banyak data berkualitas

**Q30: Bagaimana handle class imbalance saat training?**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train), 
                                      y=y_train)

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

### G. BUSINESS & USE CASE

**Q31: Siapa user aplikasi ini?**
- Government agencies (monitoring PPKM sentiment)
- Media organizations (public opinion tracking)
- Policy makers (evidence-based decisions)
- Marketing teams (brand sentiment analysis)

**Q32: Bagaimana aplikasi ini membantu pengambilan keputusan?**
- Real-time sentiment trends
- Identify issues/concerns
- Measure policy acceptance
- Early warning system untuk backlash
- Data-driven communication strategy

**Q33: Apa limitations dari aplikasi?**
- Domain-specific (hanya PPKM)
- Bahasa Indonesia only
- Tidak capture sarcasm/irony
- Membutuhui update regular untuk maintain accuracy
- Ketergantungan pada quality dataset

**Q34: Bagaimana improve aplikasi ini?**
- **Multilingual**: Support bahasa lain
- **Advanced Models**: BERT, GPT-based
- **Real-time**: Integrate dengan Twitter API
- **Explainability**: LIME/SHAP untuk interpretability
- **Multilingual**: Contextual understanding
- **Custom thresholds**: Adjust untuk recall/precision tradeoff

**Q35: ROI atau impact bisnis?**
- Faster decision making
- Cost reduction (automated monitoring)
- Better understanding of public sentiment
- Proactive crisis management
- Evidence-based policy

---

## 9️⃣ TIPS UNTUK UJIAN

### Study Guide

1. **Pahami Konsep**
   - SPK definition & components
   - ML workflow (train, eval, predict)
   - Preprocessing importance
   - Evaluation metrics meaning

2. **Practical Knowledge**
   - Bisa run `train_model.py` dan baca output
   - Pahami confusion matrix output
   - Bisa explain setiap preprocessing step
   - Tahu library & fungsinya (Sastrawi, sklearn, TfidfVectorizer)

3. **Analisis Kritis**
   - Mengapa Logistic Regression?
   - Mengapa TF-IDF?
   - Mengapa Sastrawi untuk Indonesian?
   - Trade-off precision vs recall

4. **Trouble Shooting**
   - Know error messages & solutions
   - Alternatif approaches
   - Performance improvement strategies

### Potential Question Patterns

1. **"Jelaskan..."** → Definisi + Alasan + Contoh
2. **"Bandingkan..."** → Persamaan + Perbedaan + Kapan digunakan
3. **"Bagaimana..."** → Step-by-step process
4. **"Apa masalah jika..."** → Root cause + Solutions
5. **"Rekomendasi..."** → Trade-off + Justification

### Document to Memorize

1. **Preprocessing steps**: 9 steps
2. **Evaluation metrics**: Accuracy, Precision, Recall, F1, CM
3. **TF-IDF formula**: TF × IDF concept
4. **Model comparison**: 4 models dengan kelebihan/kekurangan
5. **Data statistics**: 8,067 samples, 3 classes, 80-20 split

---

## 🔟 REFERENSI CEPAT

### File Structure
```
app.py                          → Streamlit web application
train_model.py                  → Training & evaluation
preprocessing.py                → Text preprocessing
requirements.txt                → Dependencies
data/INA_TweetsPPKM...csv       → Dataset
models/sentiment_model.pkl      → Trained model
models/vectorizer.pkl           → TF-IDF vectorizer
```

### Key Classes & Functions

**preprocessing.py**
- `TextPreprocessor`: Main class untuk preprocessing
- `clean_text()`: Remove URL, mention, punctuation, dll
- `remove_stopwords()`: Hapus stopwords
- `stem_text()`: Stemming
- `preprocess()`: Pipeline lengkap

**train_model.py**
- `SentimentClassifier`: Main class untuk training
- `load_and_preprocess_data()`: Load & clean data
- `train()`: Training process
- `evaluate()`: Evaluation metrics
- `predict()`: Prediction untuk text baru
- `save_model()`: Save ke pickle

**app.py**
- `load_model()`: Load saved model
- `predict_sentiment()`: Predict + return probability
- `plot_probabilities()`: Visualisasi

### Quick Command Reference
```bash
# Setup
pip install -r requirements.txt

# Train model
python train_model.py

# Run web app
streamlit run app.py

# Visit
http://localhost:8501
```

---

**Last Updated**: December 2024
**For**: Ujian Sistem Pendukung Keputusan (SPK)
**Duration**: ~2-3 hours study recommended

Good luck dengan ujian Anda! 🎓💪
