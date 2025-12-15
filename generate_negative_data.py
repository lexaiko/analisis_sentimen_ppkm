import pandas as pd
import random
from datetime import datetime, timedelta

# Template kalimat negatif dengan variasi
negative_templates = [
    # Tidak setuju
    "tidak setuju dengan ppkm",
    "saya tidak setuju ppkm",
    "sangat tidak setuju dengan kebijakan ppkm",
    "tidak setuju ppkm diperpanjang",
    "saya tidak setuju dengan kebijakan ppkm",
    "tidak setuju dengan aturan ppkm",
    "tidak setuju kebijakan ppkm ini",
    "tidak setuju ppkm terus menerus",
    
    # Tidak mendukung
    "tidak mendukung ppkm",
    "saya tidak mendukung kebijakan ppkm",
    "sangat tidak mendukung ppkm",
    "tidak mendukung perpanjangan ppkm",
    "tidak mendukung aturan ppkm",
    
    # Menolak
    "menolak kebijakan ppkm",
    "saya menolak ppkm",
    "menolak keras ppkm",
    "menolak perpanjangan ppkm",
    "menolak aturan ppkm yang ketat",
    
    # Merugikan
    "ppkm merugikan rakyat",
    "kebijakan ppkm sangat merugikan",
    "ppkm merugikan ekonomi",
    "ppkm merugikan pedagang kecil",
    "ppkm merugikan masyarakat",
    "sangat merugikan kebijakan ppkm",
    
    # Menyengsarakan
    "ppkm menyengsarakan rakyat",
    "ppkm menyengsarakan warga",
    "kebijakan ppkm menyengsarakan masyarakat",
    "ppkm hanya menyengsarakan rakyat kecil",
    "ppkm menyengsarakan pedagang",
    "sangat menyengsarakan ppkm ini",
    
    # Hancur/Rusak
    "ppkm bikin ekonomi hancur",
    "ekonomi hancur karena ppkm",
    "ppkm menghancurkan ekonomi rakyat",
    "usaha hancur gara gara ppkm",
    "ppkm hancurkan ekonomi",
    "ekonomi rusak karena ppkm",
    
    # Menderita
    "rakyat menderita karena ppkm",
    "ppkm membuat rakyat menderita",
    "masyarakat menderita akibat ppkm",
    "pedagang menderita karena ppkm",
    "ppkm bikin orang menderita",
    
    # Susah/Sulit
    "ppkm bikin hidup susah",
    "ppkm membuat ekonomi semakin sulit",
    "hidup makin susah karena ppkm",
    "ppkm bikin segalanya sulit",
    "susah sekali karena ppkm",
    "ppkm membuat hidup semakin sulit",
    
    # Bangkrut
    "ppkm bikin usaha bangkrut",
    "bangkrut gara gara ppkm",
    "usaha saya bangkrut karena ppkm",
    "ppkm membuat banyak usaha bangkrut",
    "pedagang bangkrut karena ppkm",
    
    # Tidak efektif
    "ppkm tidak efektif",
    "kebijakan ppkm tidak efektif sama sekali",
    "ppkm tidak berhasil",
    "ppkm gagal mengatasi pandemi",
    "ppkm tidak ada gunanya",
    
    # Kasihan/Lelah
    "kasihan rakyat karena ppkm",
    "lelah dengan ppkm",
    "capek dengan kebijakan ppkm",
    "sudah lelah dengan ppkm",
    "kapan ppkm berakhir sudah lelah",
    
    # Tidak adil
    "ppkm tidak adil",
    "kebijakan ppkm sangat tidak adil",
    "ppkm merugikan rakyat kecil saja",
    "tidak adil kebijakan ppkm ini",
    "ppkm tidak adil untuk pedagang",
    
    # Buruk
    "ppkm sangat buruk",
    "kebijakan ppkm buruk sekali",
    "ppkm berdampak buruk",
    "buruk kebijakan ppkm ini",
    "sangat buruk dampak ppkm",
    
    # Negatif umum
    "ppkm membuat lapangan kerja hilang",
    "ppkm bikin banyak orang kehilangan penghasilan",
    "ppkm membuat anak tidak bisa sekolah",
    "ppkm menambah pengangguran",
    "ppkm bikin hutang numpuk",
    "ppkm membuat usaha tutup",
    "ppkm menghilangkan mata pencaharian",
    "ppkm bikin rakyat kelaparan",
]

# Kata tambahan untuk variasi
variations = [
    "", "ini", "tersebut", "yang ada", "saat ini", "sekarang",
    "banget", "sekali", "sangat", "benar-benar", "sungguh"
]

emojis = ["", "😞", "😢", "😭", "😤", "💔", "😡"]

# Generate data
data = []
start_date = datetime(2024, 1, 1)

print("Generating 1500 negative samples...")

for i in range(1500):
    # Pilih template random
    template = random.choice(negative_templates)
    
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
        'sentiment': 2  # Negatif
    })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_file = 'data/negative_augmented_1500.csv'
df.to_csv(output_file, sep='\t', index=False)

print(f"✅ Generated {len(df)} negative samples")
print(f"✅ Saved to {output_file}")
print(f"\nSample data:")
print(df.head(10))
