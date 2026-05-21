# Prediksi Risiko Kredit Menggunakan Machine Learning

## Daftar Isi

  - Latar Belakang Proyek
  - Tujuan Bisnis
  - Dataset
  - Metodologi
  - Instalasi & Kebutuhan
  - Cara Menjalankan Proyek
  - Hasil
  - Author

## Latar Belakang Proyek

Proyek ini dikembangkan untuk klien dari industri *multifinance* yang menghadapi tantangan dalam menilai risiko kredit calon peminjam secara akurat. Penilaian risiko yang tidak tepat dapat menyebabkan kerugian finansial yang signifikan akibat pinjaman macet (*default*). Proyek ini bertujuan untuk memanfaatkan *machine learning* guna meningkatkan akurasi pengambilan keputusan pemberian pinjaman.

## Tujuan Bisnis

Tujuan utama dari proyek ini adalah untuk mengembangkan **model klasifikasi machine learning** yang mampu menjawab pertanyaan bisnis berikut:

> "Bagaimana kita dapat secara akurat memprediksi risiko kredit dari seorang peminjam berdasarkan data historis untuk meminimalkan potensi kerugian?"

Secara spesifik, model ini diharapkan dapat:

1.  **Mengoptimalkan Keputusan Bisnis:** Memberikan dasar yang kuat berbasis data untuk menyetujui atau menolak aplikasi pinjaman.
2.  **Mengurangi Potensi Kerugian:** Mengidentifikasi peminjam berisiko tinggi sejak dini untuk menghindari pinjaman yang berpotensi macet (*Charged Off*).

Variabel target didefinisikan sebagai masalah klasifikasi biner, di mana pinjaman dikategorikan sebagai:

  * **Good Credit (Risiko Rendah):** Status pinjaman `Fully Paid`.
  * **Bad Credit (Risiko Tinggi):** Status pinjaman `Charged Off`.

## Dataset

Dataset yang digunakan dalam proyek ini adalah `loan_data_2007_2014.csv`, yang berisi data historis pinjaman dari tahun 2007 hingga 2014. Dataset ini mencakup berbagai informasi mengenai pinjaman dan peminjam, seperti:

  * **Informasi Pinjaman:** `loan_amnt`, `term`, `int_rate`, `grade`, dll.
  * **Informasi Peminjam:** `emp_length`, `home_ownership`, `annual_inc`, `purpose`, dll.
  * **Riwayat Kredit:** `delinq_2yrs`, `earliest_cr_line`, `open_acc`, `pub_rec`, dll.
  * **Status Pembayaran:** `last_pymnt_d`, `total_pymnt`, dll.
  * **Variabel Target:** `loan_status`.

## Metodologi

Alur kerja proyek ini terdiri dari beberapa tahapan utama:

1.  **Data Cleaning & Preprocessing**

      * Menghapus kolom-kolom yang tidak relevan (seperti ID) dan kolom yang seluruhnya kosong.
      * Mengubah tipe data kolom agar sesuai untuk analisis (misalnya, mengubah format tanggal dan mengubah `policy_code` menjadi kategorikal).
      * Menangani nilai yang hilang (*missing values*) pada fitur-fitur penting.

2.  **Feature Engineering**

      * Membuat fitur baru `credit_history_length_month` yang dihitung dari selisih antara tanggal pinjaman disetujui (`issue_d`) dan tanggal jalur kredit pertama peminjam dibuka (`earliest_cr_line`).

3.  **Exploratory Data Analysis (EDA)**

      * Melakukan analisis univariat untuk memahami distribusi setiap fitur numerik dan kategorikal.
      * Tujuannya adalah untuk memahami data, menemukan pola, dan mengidentifikasi anomali yang akan memandu langkah persiapan data selanjutnya.

4.  **Penanganan Data Tidak Seimbang**

      * Mengatasi masalah kelas minoritas pada variabel target (pinjaman `Charged Off`) dengan menggunakan teknik *oversampling* **SMOTE** (*Synthetic Minority Over-sampling Technique*).

5.  **Pemodelan & Evaluasi**

      * Membagi data menjadi set pelatihan dan pengujian.
      * Menerapkan penskalaan fitur menggunakan `StandardScaler` untuk menyamakan rentang nilai fitur numerik.
      * Mengembangkan beberapa model klasifikasi, termasuk **Logistic Regression** dan **Random Forest Classifier**.
      * Melakukan optimasi hiperparameter pada model Random Forest menggunakan `RandomizedSearchCV` untuk mendapatkan performa terbaik.
      * Mengevaluasi performa model menggunakan metrik seperti *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *ROC-AUC Score*.

## Instalasi & Kebutuhan

Pastikan Anda memiliki Python 3.x terinstal. Kemudian, instal semua pustaka yang dibutuhkan dengan menjalankan perintah berikut:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imblearn joblib
```

Daftar pustaka utama yang digunakan:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `imblearn`
  * `joblib`

## Cara Menjalankan Proyek

1.  **Clone Repository**
    ```bash
    git clone https://github.com/akbariffianto/loan-status-classification.git
    cd loan-status-classification
    ```
2.  **Tempatkan Dataset**
    Pastikan file `loan_data_2007_2014.csv` berada di direktori yang sama dengan notebook Jupyter.
3.  **Jalankan Notebook**
    Buka dan jalankan file `FinalTask_IDX Partners_Data Scientist_Akbar Ariffianto.ipynb` menggunakan Jupyter Notebook atau JupyterLab.

## Hasil

Berdasarkan evaluasi, model **Random Forest Classifier** yang telah dioptimalkan dengan `RandomizedSearchCV` memberikan performa terbaik dalam memprediksi risiko kredit. Model ini menunjukkan keseimbangan yang baik antara metrik *precision* dan *recall* untuk kelas `Charged Off` (peminjam berisiko tinggi), yang sangat penting untuk tujuan bisnis dalam meminimalkan kerugian.

## Author

  * **Akbar Ariffianto** - [GitHub](https://www.google.com/search?q=https://github.com/akbariffianto)
