import pandas as pd
import random
from datetime import datetime, timedelta

# Template kalimat positif dengan variasi
positive_templates = [
    # Setuju
    "saya setuju dengan kebijakan ppkm",
    "setuju dengan ppkm",
    "sangat setuju dengan kebijakan ppkm",
    "saya sangat setuju ppkm dilanjutkan",
    "setuju banget sama ppkm",
    "100% setuju dengan ppkm",
    "saya setuju ppkm diperpanjang",
    "setuju dengan aturan ppkm",
    
    # Mendukung
    "saya mendukung kebijakan ppkm",
    "mendukung penuh ppkm",
    "sangat mendukung kebijakan ppkm",
    "saya mendukung ppkm untuk kebaikan bersama",
    "mendukung ppkm pemerintah",
    "full support untuk ppkm",
    "saya sangat mendukung ppkm ini",
    "mendukung kebijakan ppkm yang tepat",
    
    # Bagus/Baik
    "ppkm sangat bagus",
    "kebijakan ppkm sangat baik",
    "ppkm bagus untuk mencegah covid",
    "ppkm ini bagus sekali",
    "sangat bagus kebijakan ppkm",
    "ppkm bagus untuk kesehatan",
    "kebijakan ppkm bagus banget",
    
    # Efektif
    "ppkm efektif menurunkan kasus covid",
    "ppkm terbukti efektif",
    "kebijakan ppkm sangat efektif",
    "ppkm efektif banget",
    "sangat efektif kebijakan ppkm ini",
    "ppkm efektif untuk menekan penyebaran",
    "ppkm efektif melindungi masyarakat",
    
    # Berhasil
    "ppkm berhasil menekan kasus",
    "alhamdulillah ppkm berhasil",
    "kebijakan ppkm berhasil menurunkan angka kasus",
    "ppkm berhasil mengendalikan pandemi",
    "sangat berhasil ppkm ini",
    "ppkm berhasil melindungi warga",
    
    # Tepat/Pas
    "kebijakan ppkm sangat tepat",
    "ppkm adalah kebijakan yang tepat",
    "sangat tepat kebijakan ppkm",
    "ppkm kebijakan yang pas",
    "tepat sekali kebijakan ppkm ini",
    "ppkm solusi yang tepat",
    
    # Membantu
    "ppkm sangat membantu",
    "ppkm membantu mengatasi pandemi",
    "kebijakan ppkm sangat membantu masyarakat",
    "ppkm membantu menekan penyebaran",
    "sangat membantu kebijakan ppkm",
    "ppkm membantu melindungi kita",
    
    # Terima kasih
    "terima kasih atas kebijakan ppkm",
    "terima kasih pemerintah untuk ppkm",
    "terima kasih ppkm membuat aman",
    "makasih ya ada ppkm",
    "terima kasih kebijakan ppkm yang melindungi",
    
    # Senang/Bahagia
    "senang dengan kebijakan ppkm",
    "sangat senang ada ppkm",
    "bahagia dengan ppkm yang efektif",
    "senang sekali ppkm berhasil",
    "senang ppkm melindungi kita",
    
    # Positif umum
    "ppkm membuat situasi terkendali",
    "ppkm melindungi masyarakat",
    "ppkm menyelamatkan banyak nyawa",
    "ppkm adalah solusi terbaik",
    "ppkm membuat aman",
    "ppkm untuk keselamatan bersama",
    "ppkm langkah yang benar",
    "ppkm kebijakan bijak",
    "ppkm membawa dampak positif",
    "ppkm adalah harapan kita",
]

# Kata tambahan untuk variasi
variations = [
    "", "ini", "tersebut", "yang ada", "saat ini", "sekarang",
    "banget", "sekali", "sangat", "benar-benar", "sungguh"
]

emojis = ["", "👍", "✅", "💯", "🙏", "😊", "❤️", "🔥"]

# Generate data
data = []
start_date = datetime(2024, 1, 1)

print("Generating 2000 positive samples...")

for i in range(2000):
    # Pilih template random
    template = random.choice(positive_templates)
    
    # Tambahkan variasi
    if random.random() > 0.7:  # 30% chance add variation
        variation = random.choice(variations)
        if variation:
            template = f"{template} {variation}"
    
    # Tambahkan emoji
    if random.random() > 0.8:  # 20% chance add emoji
        emoji = random.choice(emojis)
        if emoji:
            template = f"{template} {emoji}"
    
    # Generate timestamp
    random_days = random.randint(0, 365)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    timestamp = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    
    data.append({
        'Date': timestamp.strftime('%Y-%m-%d %H:%M:%S+00:00'),
        'User': f'user_generated_{i}',
        'Tweet': template,
        'sentiment': 0  # Positif
    })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_file = 'data/positive_augmented_2000.csv'
df.to_csv(output_file, sep='\t', index=False)

print(f"✅ Generated {len(df)} positive samples")
print(f"✅ Saved to {output_file}")
print(f"\nSample data:")
print(df.head(10))
