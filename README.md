# Klasifikasi Diabetes

Proyek ini bertujuan untuk melakukan klasifikasi diabetes berdasarkan dataset yang tersedia. Dua algoritma pembelajaran mesin yang digunakan dalam proyek ini adalah **K-Nearest Neighbors (KNN)** dan **Support Vector Machine (SVM)**. Setelah membandingkan kedua algoritma, **KNN** dipilih sebagai model terbaik karena memberikan akurasi tertinggi.

## Dataset
Dataset yang digunakan dalam proyek ini berasal dari **[Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)**. Dataset ini berisi berbagai fitur medis, seperti:
- age (usia pasien)
- gender (jenis kelamin)
- BMI (indeks massa tubuh)
- hypertension (0 = tidak, 1 = ya)
- heart disease (0 = tidak, 1 = ya)
- smoking history (No Info = Tidak Ada Info, Never = Tidak Pernah, Former = Mantan, Current = Saat Ini, Not Current = Tidak Saat Ini)
- HbA1c level (rata-rata kadar gula darah dalam 2-3 bulan terakhir)
- blood glucose level (kadar gula darah saat ini)

## Instalasi dan Persyaratan
Sebelum menjalankan proyek ini, pastikan Anda memiliki **Python 3.x** dan install library yang terdapat di requirements.txt:

```bash
pip install -r requirements.txt
```

## Langkah-langkah Implementasi
1. **Import Library**: Memuat pustaka yang diperlukan.
2. **Load Dataset**: Membaca data dan melakukan eksplorasi awal.
3. **Preprocessing Data**: Menangani duplikasi data, menghapus outlier dan pembagian dataset menjadi training dan testing.
4. **Training Model**:
   - Membangun model **KNN** dan **SVM**
5. **Evaluasi Model**:
   - Menghitung metrik evaluasi seperti **akurasi, precision, recall, dan F1-score**.
   - Membandingkan performa KNN dan SVM.
6. **Kesimpulan**: Menentukan algoritma terbaik berdasarkan hasil evaluasi.

## Hasil Perbandingan Model
| Model | Akurasi |
|--------|----------|
| KNN | **93%** |
| SVM | 87% |

Berdasarkan hasil evaluasi, **KNN memberikan akurasi lebih tinggi dibandingkan SVM**, sehingga dipilih sebagai model akhir untuk klasifikasi diabetes.

## Cara Menjalankan Proyek
1. Clone repositori ini:
   ```bash
   git clone https://github.com/irsyamokta/diabetes-classification.git
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd diabetes-classification
   ```
3. Jalankan notebook Jupyter:
   ```bash
   jupyter notebook
   ```
4. Buka file `diabetes_classification.ipynb` dan jalankan setiap sel untuk melihat hasilnya.

## Deployment
Proyek ini sudah publik dan bisa diakses melalui tautan berikut https://diabete-classification.streamlit.app/

## Kontribusi
Kontribusi sangat diterima! Jika ingin menambahkan fitur atau meningkatkan model, silakan buat **pull request** atau buka **issue**.

## Author
[@irsyamokta](https://github.com/irsyamokta)
