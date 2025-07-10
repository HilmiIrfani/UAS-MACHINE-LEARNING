
# 📊 Prediksi Kategori Nilai Akhir Siswa Menggunakan Machine Learning

**Nama**: *Fathiya Hilmi Irfani*  
**NIM**: *A11.2023.15348*  
**Mata Kuliah**: *Machine Learning*  

---

## 1. 🎯 Ringkasan & Permasalahan

Nilai akhir siswa merupakan salah satu indikator penting dalam menilai performa akademik. Namun, nilai ini dipengaruhi oleh banyak faktor, mulai dari latar belakang keluarga, kebiasaan belajar, hingga kondisi sosial pribadi siswa. Proyek ini bertujuan untuk membangun model klasifikasi yang dapat memprediksi kategori nilai akhir siswa (Rendah, Sedang, Tinggi) berdasarkan data latar belakang dan perilaku mereka.

### ✅ Tujuan:

- Memprediksi nilai akhir siswa ke dalam tiga kategori: **Rendah**, **Sedang**, atau **Tinggi**
- Membandingkan performa beberapa algoritma Machine Learning
- Menyediakan antarmuka prediksi berbasis Gradio dan fungsi manual

### 🔄 Alur Penyelesaian
![alt text](https://github.com/HilmiIrfani/UAS-MACHINE-LEARNING-2025/blob/3d9413ce890b96f078ff1b1ff91f0077acd894b8/Untitled%20Diagram.drawio.png)



---

## 2. 📊 Penjelasan Dataset, EDA, dan Proses Features Dataset

Dataset yang digunakan adalah Student Performance Dataset dari UCI Machine Learning Repository, yang berisi data siswa dari dua sekolah di Portugal. Dataset ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi performa akademik siswa, khususnya pada mata pelajaran matematika.
Dataset terdiri dari 395 baris (siswa) dan 33 kolom fitur, termasuk data demografis, informasi keluarga, kebiasaan belajar, dan dua nilai ujian sebelumnya (G1, G2). Target prediksi adalah G3 (nilai akhir), yang dikategorikan menjadi:
- Diterjemahkan ke dalam Bahasa Indonesia
- Label `G3` diubah ke dalam kategori:
  - `Rendah` (< 10)
  - `Sedang` (10–14)
  - `Tinggi` (≥ 15)

### 🔧 Fitur yang Digunakan

| Tipe | Kolom | Keterangan |
|------|-------|------------|
| Numerik |Usia, Pendidikan_Ibu, Pendidikan_Ayah, Waktu_Pergi, Waktu_Belajar, Gagal_Pelajaran, Hubungan_Keluarga, Waktu_Luang, Keluar_Teman, Konsumsi_Alkohol_Harian, Konsumsi_Alkohol_Mingguan, Kesehatan, Ketidakhadiran, Nilai_1 (G1), Nilai_2 (G2) | Data numerik dari siswa |
| Kategorikal | Jenis_Kelamin, Alamat, Ukuran_Keluarga, Status_Ortu,Pekerjaan_Ibu, Pekerjaan_Ayah, Alasan_Memilih_Sekolah, Wali, Dukungan_Sekolah, Dukungan_Keluarga, Kursus_Bayar, Aktivitas_Ekstra, Pernah_Paud, Ingin_Kuliah, Ada_Internet, Hubungan_Romantis| Diubah menggunakan one-hot encoding |

**Numerik**
- Usia – umur siswa (dalam tahun)
- Pendidikan_Ibu – tingkat pendidikan ibu (0–4)
- Pendidikan_Ayah – tingkat pendidikan ayah (0–4)
- Waktu_Pergi – waktu tempuh ke sekolah (1= <15min, 4= >1jam)
- Waktu_Belajar – waktu belajar mingguan (1= <2 jam, 4= >10 jam)
- Gagal_Pelajaran – jumlah gagal mata pelajaran sebelumnya (0–3+)
- Hubungan_Keluarga – kualitas hubungan keluarga (1= sangat buruk, 5= sangat baik)
- Waktu_Luang – waktu luang setelah sekolah (1–5)
- Keluar_Teman – frekuensi keluar dengan teman (1–5)
- Konsumsi_Alkohol_Harian (Dalc) – konsumsi alkohol hari kerja (1–5)
- Konsumsi_Alkohol_Mingguan (Walc) – konsumsi alkohol akhir pekan (1–5)
- Kesehatan – status kesehatan (1= sangat buruk, 5= sangat baik)
- Ketidakhadiran – jumlah ketidakhadiran sekolah
- Nilai_1 (G1) – nilai ujian semester pertama (0–20)
- Nilai_2 (G2) – nilai ujian semester kedua (0–20)

**Kategorikal**
1. Jenis_Kelamin
F: Perempuan
M: Laki-laki

2. Alamat
U: Urban (Perkotaan)
R: Rural (Pedesaan)

3. Ukuran_Keluarga
LE3: ≤ 3 anggota keluarga
GT3: > 3 anggota keluarga

4. Status_Ortu (Pstatus)
T: Tinggal bersama
A: Terpisah

5. Pekerjaan_Ibu
at_home, health, services, teacher, other

6. Pekerjaan_Ayah
at_home, health, services, teacher, other

7. Alasan_Memilih_Sekolah (reason)
home, reputation, course, other

8. Wali (guardian)
mother, father, other

9. Dukungan_Sekolah (schoolsup)
yes atau no

10. Dukungan_Keluarga (famsup)
yes atau no

11. Kursus_Bayar (paid)
yes atau no

12. Aktivitas_Ekstra (activities)
yes atau no

13. Pernah_Paud (nursery)
yes atau no

14. Ingin_Kuliah (higher)
yes atau no

15. Ada_Internet (internet)
yes atau no

16. Hubungan_Romantis (romantic)
yes atau no

## 🎯 Pengaruh Fitur Numerik

Fitur numerik memiliki **pengaruh paling besar** terhadap hasil prediksi karena nilainya bersifat kuantitatif dan langsung mencerminkan performa akademik siswa.

| Fitur                   | Pengaruh | Penjelasan |
|------------------------|----------|------------|
| **Nilai_2 (G2)**        | 🌟🌟🌟🌟🌟 | Nilai semester kedua mendekati akhir dan sangat prediktif untuk nilai akhir. |
| **Nilai_1 (G1)**        | 🌟🌟🌟🌟 | Memberikan gambaran awal tentang kemampuan akademik siswa. |
| **Gagal_Pelajaran**     | 🌟🌟🌟 | Jumlah kegagalan mata pelajaran berbanding terbalik dengan nilai akhir. |
| **Waktu_Belajar**       | 🌟🌟🌟 | Semakin tinggi waktu belajar, semakin besar peluang nilai tinggi. |
| **Ketidakhadiran**      | 🌟🌟🌟 | Siswa dengan absensi tinggi cenderung memiliki nilai akhir yang rendah. |
| **Kesehatan**           | 🌟🌟 | Berpengaruh secara tidak langsung terhadap kehadiran dan performa belajar. |
| **Konsumsi Alkohol**    | 🌟🌟 | Konsumsi alkohol harian dan mingguan memiliki korelasi negatif dengan nilai. |

## 📊 Pengaruh Fitur Kategorikal

Fitur kategorikal memberikan **dukungan informasi tambahan**, terutama yang berkaitan dengan **dukungan sosial**, **motivasi**, dan **lingkungan belajar**.

| Fitur                      | Pengaruh | Penjelasan |
|---------------------------|----------|------------|
| **Ingin_Kuliah**           | 🌟🌟🌟 | Siswa yang ingin kuliah cenderung lebih termotivasi → nilai lebih tinggi. |
| **Dukungan_Keluarga**      | 🌟🌟🌟 | Dukungan moral dan akademik dari keluarga penting bagi kinerja belajar. |
| **Alamat (U/R)**           | 🌟🌟 | Lokasi tempat tinggal berpengaruh terhadap akses dan kenyamanan belajar. |
| **Aktivitas_Ekstra**       | 🌟🌟 | Siswa yang aktif cenderung punya keterampilan manajemen waktu lebih baik. |
| **Jenis_Kelamin**          | 🌟 | Tidak terlalu signifikan, tapi terkadang muncul perbedaan performa. |
| **Ukuran_Keluarga**        | 🌟 | Keluarga besar bisa berdampak pada distribusi perhatian/dukungan. |
| **Pekerjaan Orang Tua**    | 🌟 | Hanya sedikit pengaruh terhadap hasil akademik, kecuali guru/tenaga medis. |
| **Alasan Memilih Sekolah** | 🌟 | "Reputasi" sebagai alasan cenderung terkait dengan hasil lebih tinggi. |

---

## 🔍 Analisis Fitur dari Random Forest

Model **Random Forest** dapat menghitung **feature importance** secara langsung. Di proyek ini, hasil feature importance menunjukkan bahwa:

- **Top 5 fitur terpenting** berasal dari **fitur numerik**, yaitu:
  - Nilai_2 (G2)
  - Nilai_1 (G1)
  - Gagal_Pelajaran
  - Ketidakhadiran
  - Waktu_Belajar

## ✅ Kesimpulan

- Fitur numerik **lebih dominan** dalam menentukan prediksi.
- Fitur kategorikal memberikan **dukungan informasi kontekstual** yang membantu model membedakan siswa secara lebih realistis.
- Kombinasi keduanya menghasilkan model yang **lebih akurat dan efektif** dalam memahami karakteristik dan potensi akademik siswa.

---

## 3. 📊 Eksplorasi Data dan Feature Engineering

Beberapa proses yang dilakukan:

- Menghapus kolom `Sekolah` karena tidak relevan
- Mengubah nilai akhir `G3` menjadi kelas kategorikal
- Melakukan one-hot encoding pada semua fitur kategorikal (tanpa `drop_first`)
- Membagi dataset menjadi **80% training** dan **20% testing**

---


## 4. 🔬 Proses Learning / Modeling

### 📌 Tujuan Utama

Tujuan dari proses ini adalah membangun dan melatih model machine learning yang mampu memprediksi **kategori nilai akhir siswa** menjadi tiga kelas: `Rendah`, `Sedang`, dan `Tinggi` berdasarkan data pribadi, sosial, dan akademik siswa.

### 🧰 Langkah-Langkah Detail dalam Proses Learning

### 1️⃣ Import Library yang Dibutuhkan

Melibatkan `pandas`, `numpy`, `train_test_split`, `RandomForestClassifier`, `SVC`, `classification_report`, `f1_score`, `accuracy_score`, dan `matplotlib`, `seaborn` untuk visualisasi dan evaluasi.

### 2️⃣ Membagi Dataset Menjadi Fitur dan Label

Fitur (`X`) berisi semua kolom kecuali `Nilai_Akhir`. Label (`y`) adalah kolom target hasil kategorisasi dari `G3` menjadi Rendah, Sedang, Tinggi.

### 3️⃣ Membagi Data Latih dan Uji

`train_test_split()` digunakan dengan `test_size=0.2`, `stratify=y`, dan `random_state=42` untuk menjaga keseimbangan distribusi kelas.

### 4️⃣ Pelatihan Dua Model: SVM dan Random Forest

#### 🧠 Random Forest Classifier
Model ensemble berbasis pohon keputusan. Kuat menangani fitur campuran dan outlier.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

#### 🧠 Support Vector Machine (SVM)
Model margin optimal yang sangat baik untuk klasifikasi multikelas dan data terstruktur.

```python
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
```

### 5️⃣ Evaluasi Model

Metrik evaluasi:
- Accuracy: Persentase prediksi yang benar
- Macro F1 Score: Rata-rata F1 setiap kelas (memperhatikan keadilan antar kelas)
- Micro F1 Score: Total F1 semua kelas

Contoh penggunaan:
```python
from sklearn.metrics import classification_report, accuracy_score, f1_score
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 📈 Hasil Evaluasi

| Model         | Accuracy | Macro F1 | Micro F1 |
|---------------|----------|----------|----------|
| **SVM**       | **91.1%**| 0.924    | 0.911    |
| Random Forest | 88.6%    | 0.899    | 0.886    |

### 6️⃣ Analisis Fitur Penting

Random Forest memungkinkan kita melihat pentingnya fitur. Fitur yang paling berpengaruh:
- Nilai_2 (G2)
- Nilai_1 (G1)
- Gagal_Pelajaran
- Waktu_Belajar
- Ketidakhadiran

### 7️⃣ Visualisasi Tambahan

Dibuat grafik:
- Feature Importance
- Confusion Matrix

### ✅ Kesimpulan Modeling

- SVM menghasilkan performa terbaik dalam akurasi dan f1-score.
- Random Forest memberikan interpretasi lebih baik terhadap pentingnya fitur.
- Fitur akademik seperti `G1`, `G2`, dan `absensi` sangat krusial dalam prediksi.
- Kedua model dapat digunakan tergantung pada kebutuhan antara interpretabilitas atau akurasi maksimal.

---


## 📊 Distribusi Kategori Nilai Akhir

Gambar berikut menunjukkan **distribusi jumlah siswa berdasarkan kategori nilai akhir** dalam dataset yang digunakan.

![Distribusi Nilai Akhir](https://github.com/HilmiIrfani/UAS-MACHINE-LEARNING-2025/blob/97766956831ac0b976e1dd489bfcfaa5a747d029/Screenshot%202025-07-10%20170409.png?raw=true)

---

### 🧾 Penjelasan Grafik

- **Sumbu X**: Menampilkan tiga kelas target, yaitu `Rendah`, `Sedang`, dan `Tinggi`.
- **Sumbu Y**: Menunjukkan **jumlah siswa** dalam setiap kategori.

| Kategori Nilai Akhir | Jumlah Perkiraan | Penjelasan |
|----------------------|------------------|------------|
| **Sedang**           | ±195 siswa       | Kategori terbanyak. Mayoritas siswa berada di level ini. |
| **Rendah**           | ±130 siswa       | Jumlah signifikan, menunjukkan cukup banyak siswa dengan performa kurang baik. |
| **Tinggi**           | ±70 siswa        | Jumlah paling sedikit. Hanya sebagian kecil siswa yang mencapai hasil tinggi. |

---

### 📌 Implikasi terhadap Modeling

- Distribusi ini **tidak seimbang (imbalanced)** karena jumlah siswa dengan nilai tinggi jauh lebih sedikit dibanding kategori lain.
- Jika tidak ditangani, model machine learning cenderung akan bias ke kelas `Sedang` karena merupakan mayoritas.
- Oleh karena itu, dalam proses modeling dilakukan teknik seperti **stratified split** atau **resampling** agar model tidak berat sebelah.
- Evaluasi model dengan metrik seperti **Macro F1** juga penting agar semua kelas dinilai adil, bukan hanya akurasi total.

---

### ✅ Kesimpulan
Distribusi ini memberikan gambaran penting tentang seberapa merata performa siswa dalam dataset, dan menjadi acuan penting dalam proses preprocessing dan evaluasi model.



## 5. 💬 Diskusi Hasil dan Kesimpulan

### 🔎 Analisis

- SVM unggul dalam prediksi semua kategori, terutama pada kelas "Tinggi" yang diprediksi sempurna.
- Random Forest juga memberikan performa sangat baik dan mampu mengklasifikasi dengan F1-score tinggi.
- Fitur `Nilai_1` dan `Nilai_2` merupakan indikator kuat terhadap `Nilai_Akhir`.

### ✅ Kesimpulan

- SVM menjadi model terbaik dalam proyek ini, ideal untuk implementasi sistem prediksi nilai siswa.
- Gradio dan input manual berhasil diintegrasikan sebagai media interaktif untuk pengguna.
- Model ini dapat digunakan untuk deteksi dini siswa dengan risiko rendah, dan membantu intervensi pembelajaran yang lebih tepat sasaran.

---

## 6. 🧪 Implementasi dan Penggunaan

- Aplikasi berbasis **Gradio**: memudahkan user input data dan melihat prediksi langsung
- Fungsi input manual juga disediakan untuk penggunaan backend atau integrasi API

---

## 📌 Rekomendasi Pengembangan

- Proyek ini dapat dikembangkan lebih lanjut dengan menambahkan data kehadiran kelas, catatan psikologis, atau data nilai sebelumnya.
- Disarankan untuk melakukan **cross-validation** dan **hyperparameter tuning** agar hasil lebih optimal.
