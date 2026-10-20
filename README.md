# 💳 Pipeline Machine Learning End-to-End: Prediksi Kelayakan Kredit (Credit Approval Prediction)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-green)
![Pipeline](https://img.shields.io/badge/Pipeline-End--to--End-lightgrey)

Proyek portofolio Machine Learning yang membangun **alur kerja lengkap** untuk memprediksi apakah pengajuan kredit seorang nasabah **layak disetujui** atau **ditolak**, mulai dari data mentah, prapemrosesan, pelatihan model, evaluasi, hingga **inferensi pada data baru**.

Proyek ini terinspirasi oleh materi **"Introduction to Machine Learning"** (Ibu Taniza Tajuddin) yang menekankan alur kerja ML yang sistematis: memahami data → prapemrosesan → pembagian data latih/uji → evaluasi performa model.

> ⚠️ **Catatan:** Dataset adalah **data dummy** (25 baris) untuk demonstrasi alur kerja. Metrik yang dihasilkan bersifat ilustratif dan tidak merepresentasikan performa di dunia nyata.

---

## 📑 Daftar Isi
1. [Machine Learning Workflow](#1--machine-learning-workflow)
2. [Traditional Programming vs Machine Learning](#2--traditional-programming-vs-machine-learning)
3. [Preprocessing & Evaluation Details](#3--preprocessing--evaluation-details)
4. [Cara Menjalankan Kode](#4--cara-menjalankan-kode)
5. [Struktur Repositori](#5--struktur-repositori)
6. [Catatan Etika & Keterbatasan](#6--catatan-etika--keterbatasan)

---

## 1. 🔄 Machine Learning Workflow

```
Data Collection → Preprocessing → Train/Test Split → Model Training → Evaluation → Inference
```

| Tahap | Aktivitas pada proyek ini |
|---|---|
| **1. Data Collection** | Memuat `data_kredit_nasabah.csv` (usia, pendapatan, tanggungan, skor kredit, status pekerjaan) dan memeriksa dimensi, statistik, serta *missing value* |
| **2. Preprocessing** | Imputasi nilai kosong, *one-hot encoding* untuk status pekerjaan, standarisasi fitur numerik |
| **3. Train/Test Split** | 80% data latih dan 20% data uji (*stratified*) |
| **4. Model Training** | **Logistic Regression** (model utama) dan **Decision Tree** (pembanding), dibungkus `Pipeline` scikit-learn |
| **5. Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, dan Cross-Validation |
| **6. Inference** | Memprediksi pemohon baru beserta **peluang persetujuan** |

---

## 2. ⚖️ Traditional Programming vs Machine Learning

**Pemrograman tradisional**: manusia menulis aturan, komputer menerapkannya.

```python
# Pendekatan aturan manual (if-else)
if skor_kredit >= 650 and pendapatan >= 8_000_000:
    keputusan = "DISETUJUI"
else:
    keputusan = "DITOLAK"
```

**Machine Learning**: manusia memberi data dan jawaban historis, komputer **mempelajari polanya sendiri**.

| Aspek | Aturan Manual (If-Else) | Machine Learning |
|---|---|---|
| **Sumber aturan** | Ditentukan manusia (asumsi/intuisi) | Dipelajari dari data historis |
| **Kombinasi faktor** | Sulit menangani interaksi banyak variabel | Menimbang banyak variabel sekaligus |
| **Adaptasi** | Aturan harus diubah manual saat kondisi berubah | Cukup dilatih ulang dengan data baru |
| **Keluaran** | Ya/tidak kaku | Ya/tidak **plus peluang** (mis. 82% layak) |
| **Transparansi** | Sangat jelas | Perlu interpretasi (mis. koefisien model) |
| **Syarat** | Tidak butuh data | Butuh data berkualitas |

> 💡 Dalam praktik, perusahaan sering **menggabungkan keduanya**: ML memberi skor risiko, sementara aturan kebijakan (batas minimum, regulasi) tetap menjadi pagar pengaman.

---

## 3. 🧹 Preprocessing & Evaluation Details

### Mengapa pembersihan data penting?
**Garbage in, garbage out.** Model hanya sebaik data yang dipelajarinya.

| Langkah | Alasan |
|---|---|
| **Imputasi *missing value*** (median untuk numerik, modus untuk kategorikal) | Banyak algoritma tidak dapat memproses nilai kosong. Median dipilih karena tahan terhadap *outlier* pendapatan |
| **One-Hot Encoding** | Status pekerjaan bersifat nominal (tanpa urutan). Label 0/1/2 akan menyiratkan urutan palsu |
| **Standarisasi** | Pendapatan (jutaan rupiah) dan tanggungan (0–5) berbeda skala jauh. Tanpa standarisasi, fitur berskala besar mendominasi |
| **Split sebelum preprocessing** | Mencegah *data leakage*: statistik data uji tidak boleh mempengaruhi pelatihan |
| **Pipeline** | Memastikan transformasi yang sama diterapkan pada data latih, uji, dan data baru |

### Mengapa Precision & Recall krusial pada risiko finansial?
**Accuracy saja menyesatkan**, terutama bila kelas tidak seimbang. Pada kasus kredit, **kedua jenis kesalahan punya biaya berbeda**:

| Kesalahan | Arti | Dampak bisnis |
|---|---|---|
| **False Positive** | Nasabah tidak layak **disetujui** | 🔴 Risiko kredit macet, kerugian langsung |
| **False Negative** | Nasabah layak **ditolak** | 🟡 Kehilangan peluang pendapatan bunga & loyalitas nasabah |

| Metrik | Pertanyaan yang dijawab | Relevansi |
|---|---|---|
| **Precision** (kelas Layak) | Dari semua yang disetujui model, berapa yang benar-benar layak? | **Pelindung dari kredit macet** |
| **Recall** (kelas Layak) | Dari semua nasabah layak, berapa yang berhasil dikenali? | Mengukur peluang bisnis yang tidak terlewat |
| **F1-Score** | Keseimbangan Precision dan Recall | Ringkasan saat keduanya penting |
| **Confusion Matrix** | Rincian TP, TN, FP, FN | Menunjukkan jenis kesalahan secara eksplisit |

Lembaga yang sangat menghindari kredit macet biasanya mengutamakan **Precision**. Lembaga yang ingin memperluas penyaluran kredit memperhatikan **Recall**. Ambang keputusan (*threshold*) dapat disesuaikan dengan profil risiko lembaga.

---

## 4. ▶️ Cara Menjalankan Kode

```bash
# 1. Clone repositori
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan pipeline
python ml_credit_pipeline.py
```

**Keluaran:** ringkasan data, hasil cross-validation, metrik evaluasi dan confusion matrix kedua model, koefisien model, **hasil inferensi untuk 3 pemohon baru**, serta file `model_kredit.joblib`.

---

## 5. 📁 Struktur Repositori

```
├── data_kredit_nasabah.csv     # Dataset dummy (25 baris)
├── ml_credit_pipeline.py       # Pipeline ML end-to-end
├── model_kredit.joblib         # Model tersimpan (dibuat saat skrip dijalankan)
├── requirements.txt            # Dependensi Python
└── README.md                   # Dokumentasi proyek
```

---

## 6. 🧭 Catatan Etika & Keterbatasan

- **Data sangat kecil (25 baris):** data uji hanya 5 baris, sehingga metrik sangat bervariasi. Gunakan data lebih besar untuk kesimpulan yang andal.
- **Keadilan (*fairness*):** pada sistem kredit nyata, penggunaan fitur seperti usia perlu ditinjau agar tidak menimbulkan diskriminasi, serta mengikuti regulasi dan perlindungan data pribadi yang berlaku.
- **Keputusan akhir:** model bersifat **alat bantu**, keputusan kredit tetap perlu pertimbangan manusia.

## 🚀 Pengembangan Lanjutan
- Tambahkan model **Random Forest / Gradient Boosting** dan *hyperparameter tuning*.
- Tangani ketidakseimbangan kelas (*class weight*, SMOTE).
- Tambahkan kurva **ROC-AUC** dan analisis *threshold*.
- Bangun antarmuka sederhana (Streamlit / FastAPI) untuk inferensi.

---

👤 **Author:** <Nama Anda> | 🔗 [LinkedIn](#) | 📧 email@contoh.com
