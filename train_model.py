import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import TextPreprocessor

# Buat folder models jika belum ada
os.makedirs('models', exist_ok=True)

class SentimentClassifier:
    def __init__(self, model_type='logistic'):
        """
        model_type: 'naive_bayes', 'logistic', 'svm', 'random_forest'
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        # Disable stemming untuk performa lebih cepat (stemming sangat lambat untuk dataset besar)
        self.preprocessor = TextPreprocessor(use_stemming=False)
        
        # Inisialisasi model
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type tidak valid")
    
    def load_and_preprocess_data(self, filepath, augmented_filepaths=None):
        """
        Load dan preprocess data
        """
        print("Loading data...")
        df = pd.read_csv(filepath, sep='\t')
        
        # Load augmented data jika ada (bisa multiple files)
        if augmented_filepaths:
            if isinstance(augmented_filepaths, str):
                augmented_filepaths = [augmented_filepaths]
            
            for aug_file in augmented_filepaths:
                if os.path.exists(aug_file):
                    print(f"Loading augmented data from {aug_file}...")
                    df_aug = pd.read_csv(aug_file, sep='\t')
                    df = pd.concat([df, df_aug], ignore_index=True)
                    print(f"  Added {len(df_aug)} samples")
        
        print(f"Total data: {len(df)}")
        print(f"Distribusi label:\n{df['sentiment'].value_counts()}\n")
        
        print("Preprocessing data...")
        df_clean = self.preprocessor.preprocess_dataframe(df)
        
        print(f"Data setelah preprocessing: {len(df_clean)}")
        print(f"Distribusi label setelah preprocessing:\n{df_clean['sentiment'].value_counts()}\n")
        
        return df_clean
    
    def balance_data(self, df, strategy='combined'):
        """
        Balance data dengan undersampling dan/atau oversampling
        strategy: 'undersample', 'oversample', 'combined'
        """
        print(f"Balancing data dengan strategy: {strategy}")
        
        # Pisahkan berdasarkan label
        df_pos = df[df['sentiment'] == 0]  # Positif
        df_net = df[df['sentiment'] == 1]  # Netral
        df_neg = df[df['sentiment'] == 2]  # Negatif
        
        print(f"Sebelum balancing:")
        print(f"  Positif : {len(df_pos)}")
        print(f"  Netral  : {len(df_net)}")
        print(f"  Negatif : {len(df_neg)}")
        
        if strategy == 'undersample':
            # Undersample netral ke jumlah rata-rata
            target_size = (len(df_pos) + len(df_neg)) // 2
            df_net_balanced = df_net.sample(n=min(target_size * 2, len(df_net)), random_state=42)
            df_balanced = pd.concat([df_pos, df_net_balanced, df_neg], ignore_index=True)
            
        elif strategy == 'oversample':
            # Oversample positif dan negatif
            target_size = len(df_net)
            df_pos_balanced = df_pos.sample(n=target_size, replace=True, random_state=42)
            df_neg_balanced = df_neg.sample(n=target_size, replace=True, random_state=42)
            df_balanced = pd.concat([df_pos_balanced, df_net, df_neg_balanced], ignore_index=True)
            
        elif strategy == 'combined':
            # Kombinasi: undersample netral + oversample positif & negatif
            # Target: buat distribusi lebih seimbang
            target_size = max(len(df_pos), len(df_neg)) * 2
            
            # Undersample netral
            df_net_balanced = df_net.sample(n=min(target_size, len(df_net)), random_state=42)
            
            # Oversample positif dan negatif
            df_pos_balanced = df_pos.sample(n=target_size, replace=True, random_state=42)
            df_neg_balanced = df_neg.sample(n=target_size, replace=True, random_state=42)
            
            df_balanced = pd.concat([df_pos_balanced, df_net_balanced, df_neg_balanced], ignore_index=True)
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nSetelah balancing:")
        print(f"  Positif : {len(df_balanced[df_balanced['sentiment'] == 0])}")
        print(f"  Netral  : {len(df_balanced[df_balanced['sentiment'] == 1])}")
        print(f"  Negatif : {len(df_balanced[df_balanced['sentiment'] == 2])}")
        print(f"  Total   : {len(df_balanced)}\n")
        
        return df_balanced
    
    def train(self, X_train, y_train):
        """
        Training model
        """
        print(f"Training model {self.model_type}...")
        
        # Fit vectorizer dan transform data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_tfidf, y_train)
        
        print("Training selesai!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluasi model
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        target_names = ['Positif (0)', 'Netral (1)', 'Negatif (2)']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return accuracy, y_pred, cm
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Positif', 'Netral', 'Negatif'],
                    yticklabels=['Positif', 'Netral', 'Negatif'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nConfusion matrix saved to {save_path}")
    
    def predict(self, text):
        """
        Prediksi sentimen untuk teks baru
        """
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Transform
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get probability jika model support
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_tfidf)[0]
        else:
            probabilities = None
        
        return prediction, probabilities
    
    def save_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """
        Simpan model dan vectorizer
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """
        Load model dan vectorizer
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print("Model dan vectorizer berhasil di-load!")

def compare_models(df, test_size=0.2):
    """
    Bandingkan performa berbagai model
    """
    X = df['cleaned_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models = ['naive_bayes', 'logistic', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print('='*50)
        
        classifier = SentimentClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        accuracy, y_pred, cm = classifier.evaluate(X_test, y_test)
        
        results[model_type] = accuracy
        
        # Plot confusion matrix
        classifier.plot_confusion_matrix(cm, f'models/{model_type}_confusion_matrix.png')
    
    print(f"\n{'='*50}")
    print("SUMMARY - Model Comparison")
    print('='*50)
    for model, acc in results.items():
        print(f"{model:20s}: {acc:.4f}")
    
    return results

if __name__ == "__main__":
    # Load dan preprocess data
    classifier = SentimentClassifier(model_type='logistic')
    df = classifier.load_and_preprocess_data(
        'data/INA_TweetsPPKM_Labeled_Pure.csv',
        augmented_filepaths=[
            'data/augmented_data.csv',
            'data/positive_augmented_2000.csv',
            'data/negative_augmented_1500.csv'
        ]
    )
    
    # Balance data untuk mengatasi class imbalance
    df_balanced = classifier.balance_data(df, strategy='combined')
    
    # Split data
    X = df_balanced['cleaned_text']
    y = df_balanced['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    print(f"\n{'='*50}")
    print("TRAINING BEST MODEL (Logistic Regression)")
    print('='*50)
    classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy, y_pred, cm = classifier.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(cm, 'models/confusion_matrix.png')
    
    # Save model
    classifier.save_model()
    
    # Test prediction
    print(f"\n{'='*50}")
    print("TEST PREDICTIONS")
    print('='*50)
    
    test_texts = [
        "PPKM sangat bagus untuk mencegah penyebaran covid",
        "PPKM membuat ekonomi semakin sulit dan rakyat menderita",
        "Pemerintah mengumumkan perpanjangan PPKM level 2"
    ]
    
    for text in test_texts:
        pred, proba = classifier.predict(text)
        sentiment_label = ['Positif', 'Netral', 'Negatif'][pred]
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment_label} ({pred})")
        if proba is not None:
            print(f"Probabilities: Positif={proba[0]:.3f}, Netral={proba[1]:.3f}, Negatif={proba[2]:.3f}")
    
    # Compare all models (opsional - uncomment jika ingin membandingkan)
    # print(f"\n{'='*50}")
    # print("COMPARING ALL MODELS")
    # print('='*50)
    # compare_models(df)
