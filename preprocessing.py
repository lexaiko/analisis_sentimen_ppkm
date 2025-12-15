import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self, use_stemming=False):
        # Inisialisasi stemmer dan stopword remover
        self.use_stemming = use_stemming
        
        if use_stemming:
            factory_stemmer = StemmerFactory()
            self.stemmer = factory_stemmer.create_stemmer()
        else:
            self.stemmer = None
        
        # Kata-kata penting yang tidak boleh dihapus (negasi, intensifier, dll)
        self.important_words = {
            'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 
            'kurang', 'sangat', 'sekali', 'amat', 'terlalu',
            'lebih', 'paling', 'hanya', 'saja', 'bahkan'
        }
        
        # Get stopwords dari Sastrawi
        factory_stopword = StopWordRemoverFactory()
        self.base_stopwords = set(factory_stopword.get_stop_words())
        
        # Remove kata-kata penting dari stopword list
        self.stopwords = self.base_stopwords - self.important_words
        
    def clean_text(self, text):
        """
        Membersihkan teks dari URL, mention, hashtag, angka, dan karakter khusus
        """
        if pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Hapus mention (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Hapus hashtag (#)
        text = re.sub(r'#\w+', '', text)
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus tanda baca dan karakter khusus
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Hapus whitespace berlebih
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Menghapus stopwords dari teks, tapi tetap pertahankan kata-kata penting
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """
        Melakukan stemming pada teks
        """
        if self.use_stemming and self.stemmer:
            return self.stemmer.stem(text)
        return text
    
    def preprocess(self, text):
        """
        Pipeline lengkap untuk preprocessing teks
        """
        # Bersihkan teks
        text = self.clean_text(text)
        
        # Hapus stopwords
        text = self.remove_stopwords(text)
        
        # Stemming
        text = self.stem_text(text)
        
        return text
    
    def preprocess_dataframe(self, df, text_column='Tweet'):
        """
        Preprocessing untuk dataframe
        """
        df = df.copy()
        print(f"Processing {len(df)} tweets...")
        df['cleaned_text'] = df[text_column].apply(self.preprocess)
        
        # Hapus baris dengan teks kosong setelah preprocessing
        df = df[df['cleaned_text'].str.strip() != '']
        
        return df

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    # Load data
    df = pd.read_csv('data/INA_TweetsPPKM_Labeled_Pure.csv', sep='\t')
    print(f"Total data sebelum preprocessing: {len(df)}")
    
    # Preprocess
    df_clean = preprocessor.preprocess_dataframe(df)
    print(f"Total data setelah preprocessing: {len(df_clean)}")
    
    # Tampilkan contoh
    print("\nContoh preprocessing:")
    for i in range(3):
        print(f"\nOriginal: {df.iloc[i]['Tweet'][:100]}...")
        print(f"Cleaned: {df_clean.iloc[i]['cleaned_text'][:100]}...")
    
    # Simpan hasil preprocessing
    df_clean.to_csv('data/preprocessed_tweets.csv', index=False)
    print("\nData tersimpan di data/preprocessed_tweets.csv")
