import streamlit as st
import pickle
import pandas as pd
from preprocessing import TextPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Sentimen PPKM",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .positif {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .netral {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .negatif {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model dan vectorizer"""
    try:
        with open('models/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        # Disable stemming untuk performa lebih cepat
        preprocessor = TextPreprocessor(use_stemming=False)
        return model, vectorizer, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Pastikan Anda sudah menjalankan train_model.py untuk melatih model!")
        return None, None, None

def predict_sentiment(text, model, vectorizer, preprocessor):
    """Prediksi sentimen dari teks"""
    # Preprocess text
    cleaned_text = preprocessor.preprocess(text)
    
    # Transform ke TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Prediksi
    prediction = model.predict(text_tfidf)[0]
    
    # Probabilitas
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
    else:
        probabilities = None
    
    return prediction, probabilities, cleaned_text

def plot_probabilities(probabilities):
    """Plot probabilitas dalam bentuk bar chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    labels = ['Positif (0)', 'Netral (1)', 'Negatif (2)']
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    bars = ax.barh(labels, probabilities, color=colors, alpha=0.7)
    
    # Tambahkan nilai di setiap bar
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', 
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilitas', fontsize=12)
    ax.set_title('Distribusi Probabilitas Sentimen', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.title("💬 Klasifikasi Sentimen Tweet PPKM")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        st.markdown("""
        Aplikasi ini mengklasifikasikan sentimen tweet tentang PPKM menjadi 3 kategori:
        
        - **0 = Positif** 😊
        - **1 = Netral** 😐
        - **2 = Negatif** 😞
        
        ### Cara Penggunaan:
        1. Masukkan komentar/tweet di text area
        2. Klik tombol "Analisis Sentimen"
        3. Lihat hasil klasifikasi dan probabilitas
        
        ### Model:
        - Algoritma: Logistic Regression
        - Features: TF-IDF (max 5000 features)
        - Dataset: Twitter PPKM Indonesia
        """)
        
        st.markdown("---")
        st.markdown("**Dibuat dengan ❤️ menggunakan Streamlit**")
    
    # Load model
    model, vectorizer, preprocessor = load_model()
    
    if model is None:
        st.warning("⚠️ Model belum tersedia. Jalankan `train_model.py` terlebih dahulu!")
        st.code("python train_model.py", language="bash")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Komentar")
        
        # Text input
        user_input = st.text_area(
            "Masukkan komentar atau tweet tentang PPKM:",
            height=150,
            placeholder="Contoh: PPKM sangat membantu menurunkan kasus COVID-19 di Indonesia",
            help="Ketik atau paste komentar yang ingin dianalisis"
        )
        
        # Tombol analisis
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.subheader("📊 Contoh Input")
        st.markdown("""
        **Positif:**
        - "PPKM efektif menurunkan kasus COVID"
        
        **Netral:**
        - "Pemerintah umumkan perpanjangan PPKM"
        
        **Negatif:**
        - "PPKM bikin ekonomi hancur"
        """)
    
    # Clear button logic
    if clear_btn:
        st.rerun()
    
    # Analisis sentimen
    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Menganalisis sentimen..."):
                # Prediksi
                prediction, probabilities, cleaned_text = predict_sentiment(
                    user_input, model, vectorizer, preprocessor
                )
                
                # Mapping sentimen
                sentiment_map = {
                    0: ("Positif", "positif", "😊", "#d4edda"),
                    1: ("Netral", "netral", "😐", "#fff3cd"),
                    2: ("Negatif", "negatif", "😞", "#f8d7da")
                }
                
                sentiment_label, sentiment_class, emoji, color = sentiment_map[prediction]
                
                st.markdown("---")
                st.subheader("📊 Hasil Analisis")
                
                # Hasil prediksi
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    st.markdown(f"""
                    <div class="sentiment-box {sentiment_class}">
                        {emoji} SENTIMEN: {sentiment_label.upper()} ({prediction})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detail teks
                    with st.expander("🔍 Detail Preprocessing"):
                        st.markdown("**Teks Original:**")
                        st.info(user_input)
                        st.markdown("**Teks Setelah Preprocessing:**")
                        st.success(cleaned_text if cleaned_text else "*[Teks kosong setelah preprocessing]*")
                
                with col_result2:
                    # Confidence
                    if probabilities is not None:
                        confidence = probabilities[prediction] * 100
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.1f}%",
                            delta=None
                        )
                
                # Plot probabilitas
                if probabilities is not None:
                    st.markdown("### 📈 Distribusi Probabilitas")
                    fig = plot_probabilities(probabilities)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Tabel probabilitas
                    prob_df = pd.DataFrame({
                        'Sentimen': ['Positif (0)', 'Netral (1)', 'Negatif (2)'],
                        'Probabilitas': [f"{p:.2%}" for p in probabilities],
                        'Nilai': probabilities
                    })
                    
                    with st.expander("📋 Lihat Detail Probabilitas"):
                        st.dataframe(
                            prob_df[['Sentimen', 'Probabilitas']],
                            hide_index=True,
                            use_container_width=True
                        )
                
                # Interpretasi
                st.markdown("### 💡 Interpretasi")
                if prediction == 0:
                    st.success("""
                    **Sentimen Positif** menunjukkan bahwa komentar mengandung opini yang mendukung 
                    atau memberikan penilaian baik terhadap PPKM.
                    """)
                elif prediction == 1:
                    st.info("""
                    **Sentimen Netral** menunjukkan bahwa komentar bersifat informatif atau tidak 
                    mengandung opini yang jelas (baik positif maupun negatif).
                    """)
                else:
                    st.error("""
                    **Sentimen Negatif** menunjukkan bahwa komentar mengandung kritik atau 
                    penilaian buruk terhadap PPKM.
                    """)
    
    # Batch prediction
    st.markdown("---")
    st.subheader("📦 Analisis Batch (Multiple Comments)")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom 'text' atau 'Tweet'",
        type=['csv'],
        help="File CSV harus memiliki kolom bernama 'text' atau 'Tweet'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Deteksi kolom text
            text_col = None
            if 'text' in df.columns:
                text_col = 'text'
            elif 'Tweet' in df.columns:
                text_col = 'Tweet'
            else:
                st.error("File harus memiliki kolom 'text' atau 'Tweet'")
                return
            
            st.write(f"📊 Total data: {len(df)}")
            
            if st.button("🚀 Proses Batch", type="primary"):
                with st.spinner("Memproses batch predictions..."):
                    predictions = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, text in enumerate(df[text_col]):
                        pred, proba, _ = predict_sentiment(text, model, vectorizer, preprocessor)
                        predictions.append(pred)
                        
                        if proba is not None:
                            confidences.append(proba[pred])
                        else:
                            confidences.append(None)
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    df['prediction'] = predictions
                    df['sentiment_label'] = df['prediction'].map({
                        0: 'Positif', 1: 'Netral', 2: 'Negatif'
                    })
                    if confidences[0] is not None:
                        df['confidence'] = confidences
                    
                    st.success("✅ Batch processing selesai!")
                    
                    # Tampilkan hasil
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    # Statistik
                    st.markdown("### 📊 Statistik Hasil")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    sentiment_counts = df['sentiment_label'].value_counts()
                    
                    with col_stat1:
                        st.metric("😊 Positif", sentiment_counts.get('Positif', 0))
                    with col_stat2:
                        st.metric("😐 Netral", sentiment_counts.get('Netral', 0))
                    with col_stat3:
                        st.metric("😞 Negatif", sentiment_counts.get('Negatif', 0))
                    
                    # Download hasil
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Hasil (CSV)",
                        data=csv,
                        file_name=f"hasil_sentimen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
