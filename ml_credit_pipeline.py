"""
Pipeline Machine Learning End-to-End: Prediksi Kelayakan Kredit
================================================================
Alur: Data Collection -> Preprocessing -> Train/Test Split ->
      Model Training -> Evaluation -> Inference (data baru)

Target: Status_Kelayakan (1 = Layak/Approved, 0 = Tidak Layak/Rejected)
"""

# ---------------------------------------------------------------
# 1. IMPORT LIBRARY
# ---------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42  # agar hasil dapat direproduksi

# ---------------------------------------------------------------
# 2. MEMUAT DATASET (Data Collection & Understanding)
# ---------------------------------------------------------------
df = pd.read_csv("data_kredit_nasabah.csv")

print("=== 5 Baris Pertama ===")
print(df.head())
print("\n=== Dimensi Data (baris, kolom) ===", df.shape)
print("\n=== Statistik Deskriptif ===")
print(df.describe().round(1))
print("\n=== Jumlah Missing Value per Kolom ===")
print(df.isnull().sum())
print("\n=== Distribusi Target (0 = Tidak Layak, 1 = Layak) ===")
print(df["Status_Kelayakan"].value_counts())

# ---------------------------------------------------------------
# 3. PEMILAHAN FITUR & TARGET
# ---------------------------------------------------------------
# ID_Nasabah hanya pengenal unik (tidak punya makna prediktif) -> dibuang.
X = df.drop(columns=["ID_Nasabah", "Status_Kelayakan"])
y = df["Status_Kelayakan"]

fitur_numerik = ["Usia", "Pendapatan_Bulanan", "Jumlah_Tanggungan", "Skor_Kredit_Historis"]
fitur_kategorikal = ["Status_Pekerjaan"]

# ---------------------------------------------------------------
# 4. PEMBAGIAN DATA: 80% TRAINING, 20% TESTING
# ---------------------------------------------------------------
# Pembagian dilakukan SEBELUM preprocessing agar statistik data uji tidak
# "bocor" ke proses pelatihan (data leakage).
# stratify=y menjaga proporsi layak/tidak layak tetap sama di kedua set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nData latih: {len(X_train)} baris | Data uji: {len(X_test)} baris")

# ---------------------------------------------------------------
# 5. DATA PREPROCESSING (dalam ColumnTransformer)
# ---------------------------------------------------------------
# Numerik   : isi missing value dengan MEDIAN (tahan terhadap outlier),
#             lalu STANDARISASI (rata-rata 0, std 1) agar skala fitur setara
#             (Pendapatan dalam jutaan vs Jumlah_Tanggungan 0-5).
# Kategorikal: isi missing value dengan MODUS, lalu ONE-HOT ENCODING.
#             One-hot dipilih (bukan LabelEncoder) karena Status_Pekerjaan
#             bersifat nominal; label 0/1/2 akan menyiratkan urutan palsu.
transformer_numerik = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
transformer_kategorikal = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),  # aman jika ada kategori baru
])
preprocessor = ColumnTransformer([
    ("num", transformer_numerik, fitur_numerik),
    ("cat", transformer_kategorikal, fitur_kategorikal),
])

# ---------------------------------------------------------------
# 6. PELATIHAN MODEL (Supervised Learning)
# ---------------------------------------------------------------
# Model utama : Logistic Regression (sederhana, stabil, mudah dijelaskan).
# Pembanding  : Decision Tree (untuk melihat perbandingan pendekatan).
# Dibungkus Pipeline: preprocessing di-fit HANYA pada data latih, lalu
# otomatis diterapkan ke data uji dan data baru saat prediksi.
model_utama = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
model_pembanding = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
])

model_utama.fit(X_train, y_train)
model_pembanding.fit(X_train, y_train)

# Data uji hanya 5 baris -> tambahkan cross-validation pada data latih
# sebagai gambaran stabilitas yang lebih andal.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for nama, mdl in [("Logistic Regression", model_utama), ("Decision Tree", model_pembanding)]:
    skor_cv = cross_val_score(mdl, X_train, y_train, cv=cv, scoring="f1")
    print(f"CV F1 (5-fold) {nama:<20}: {skor_cv.mean():.2f} (+/- {skor_cv.std():.2f})")

# ---------------------------------------------------------------
# 7. EVALUASI MODEL
# ---------------------------------------------------------------
def evaluasi(nama, model):
    """Cetak metrik lengkap pada data uji (kelas positif = 1 / Layak)."""
    y_pred = model.predict(X_test)
    print(f"\n{'=' * 55}\nEVALUASI: {nama}\n{'=' * 55}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.2f}")

    # Confusion matrix: baris = nilai aktual, kolom = nilai prediksi
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Aktual: Tidak Layak", "Aktual: Layak"],
        columns=["Prediksi: Tidak Layak", "Prediksi: Layak"],
    )
    print("\nConfusion Matrix:")
    print(cm_df)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN={tn} | FP={fp} (kredit disetujui padahal tidak layak -> risiko macet)")
    print(f"FN={fn} (nasabah layak ditolak -> peluang bisnis hilang) | TP={tp}")

    print("\nClassification Report (per kelas):")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Tidak Layak", "Layak"], zero_division=0,
    ))

evaluasi("Logistic Regression (model utama)", model_utama)
evaluasi("Decision Tree (pembanding)", model_pembanding)

# Interpretasi model utama: koefisien positif -> meningkatkan peluang disetujui
nama_fitur = model_utama.named_steps["prep"].get_feature_names_out()
koefisien = pd.Series(model_utama.named_steps["model"].coef_[0], index=nama_fitur)
print("\n=== Koefisien Logistic Regression (urut berpengaruh) ===")
print(koefisien.sort_values(ascending=False).round(3))

# ---------------------------------------------------------------
# 8. INFERENSI PADA DATA BARU (UNSEEN DATA)
# ---------------------------------------------------------------
# Tiga pemohon baru yang tidak ada di dataset. Pemohon ke-3 memiliki
# nilai kosong untuk membuktikan pipeline menangani missing value saat inferensi.
data_baru = pd.DataFrame([
    {"Usia": 35, "Pendapatan_Bulanan": 14000000, "Jumlah_Tanggungan": 2,
     "Skor_Kredit_Historis": 730, "Status_Pekerjaan": "Tetap"},
    {"Usia": 24, "Pendapatan_Bulanan": 4000000, "Jumlah_Tanggungan": 1,
     "Skor_Kredit_Historis": 550, "Status_Pekerjaan": "Kontrak"},
    {"Usia": 40, "Pendapatan_Bulanan": 9000000, "Jumlah_Tanggungan": 3,
     "Skor_Kredit_Historis": np.nan, "Status_Pekerjaan": "Wiraswasta"},
])

prediksi = model_utama.predict(data_baru)
peluang_layak = model_utama.predict_proba(data_baru)[:, 1]

hasil_inferensi = data_baru.copy()
hasil_inferensi["Peluang_Layak"] = (peluang_layak * 100).round(1).astype(str) + "%"
hasil_inferensi["Keputusan"] = np.where(prediksi == 1, "DISETUJUI", "DITOLAK")

print("\n=== HASIL INFERENSI PADA DATA BARU ===")
print(hasil_inferensi.to_string(index=False))

# ---------------------------------------------------------------
# 9. SIMPAN MODEL (opsional, untuk deployment)
# ---------------------------------------------------------------
joblib.dump(model_utama, "model_kredit.joblib")
print("\nModel disimpan ke 'model_kredit.joblib'")

# Catatan: dataset hanya 25 baris (dummy), sehingga metrik bersifat ilustratif
# untuk mendemonstrasikan alur kerja, bukan bukti performa di dunia nyata.
