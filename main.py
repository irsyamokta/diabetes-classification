import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from collections import Counter

# Set page title
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# Dataset
DATASET_PATH = "./dataset/diabetes_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# Model
MODEL_KNN_PATH = "./model/KNN_best_model.pkl"
MODEL_SVM_PATH = "./model/SVM_model.pkl"
model_knn = joblib.load(MODEL_KNN_PATH)
model_svm = joblib.load(MODEL_SVM_PATH)

# Label Encoder
LABEL_ENCODER_PATH = "./utils/label_encoder.pkl"
label_encoders = joblib.load(LABEL_ENCODER_PATH)

def remove_duplicates(data):
    """Menghapus data duplikat."""
    before_dedup = len(data)
    data.drop_duplicates(inplace=True)
    after_dedup = len(data)
    return data, before_dedup, after_dedup

def remove_outliers_iqr(data, columns):
    """Menghapus outlier berdasarkan metode IQR."""
    df_cleaned = data.copy()
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned

def encode_categorical(data, categorical_columns):
    """Melakukan encoding pada variabel kategori."""
    label_encoders = {}
    df_encoded = data.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    return df_encoded, label_encoders


def oversample_smote(X, y):
    """Menyeimbangkan dataset menggunakan SMOTE."""
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def split_data(X, y, test_size=0.2, random_state=42):
    """Membagi dataset menjadi Train dan Test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Custom CSS for sidebar animation
st.markdown(
    """
    <style>
        .sidebar .block-container { padding-top: 0px; }
        .sidebar-button { 
            padding: 12px; 
            margin: 8px 0; 
            border-radius: 8px; 
            transition: background-color 0.3s ease, transform 0.2s ease; 
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            display: block;
            text-align: center;
        }
        .sidebar-button:hover { 
            background-color: #f0f0f0; 
            transform: scale(1.05); 
        }
        .sidebar-selected {
            background-color: #4CAF50;
            color: white;
        }
        
        div.stButton > button {
            margin-top: 10px;
            width: 100%;
            height: 40px;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("Klasifikasi Diabetes Berdasarkan Faktor Medis Pasien")
    
    stages = [
        "Business Understanding", 
        "Data Understanding", 
        "Data Preparation", 
        "Modeling", 
        "Evaluation", 
        "Deployment"
    ]
    
    selected_stage = st.session_state.get("selected_stage", "Business Understanding")
    
    for stage in stages:
        if st.button(stage, key=stage, help=f"Go to {stage}", use_container_width=True):
            st.session_state.selected_stage = stage
            selected_stage = stage

# Content based on selection
if selected_stage == "Business Understanding":
    st.header("Business Understanding")
    st.write("Pada tahap pertama, kami memahami permasalahan bisnis yang ingin diselesaikan menggunakan Machine Learning. Dalam permasalahan ini, tujuan utamanya adalah membangun model yang dapat mengidentifikasi individu dengan risiko diabetes berdasarkan faktor kesehatan mereka. Tahapannya melibatkan identifikasi tujuan bisnis, memahami kebutuhan stakeholders, serta menentukan metrik keberhasilan model, seperti akurasi atau F1-score.")
    
    st.subheader("1️⃣ Define Business Objectives")
    st.write("""
    **Tujuan utama:**
        Membangun model machine learning untuk mengklasifikasi risiko diabetes berdasarkan faktor medis pasien.

    **Tujuan spesifik:**
    - Membantu tenaga medis dalam mendeteksi dini diabetes pada pasien berdasarkan riwayat kesehatan.
    - Mengurangi jumlah kasus diabetes yang tidak terdiagnosis dengan memberikan rekomendasi berdasarkan klasifikasi model.
    - Meningkatkan efektivitas dalam perencanaan pengobatan dan intervensi dini.
    """)
    
    st.subheader("2️⃣ Assess Current Situation")
    st.write("""
    **Analisis kondisi saat ini:**
    - Diabetes merupakan penyakit kronis yang berkembang secara perlahan dan sering kali tidak terdeteksi hingga mencapai tahap lanjut.
    - Faktor risiko utama meliputi usia, hipertensi, penyakit jantung, BMI, dan kadar gula darah.
    - Diagnosis diabetes biasanya dilakukan melalui tes HbA1c dan tes kadar glukosa darah yang membutuhkan biaya dan waktu.
    - Jika dapat diidentifikasi lebih awal, pasien berisiko tinggi dapat diberi intervensi lebih cepat melalui pola makan, olahraga, dan terapi medis.

    **Masukan dari stakeholder:**
    - **Dokter & tenaga medis:** Memerlukan alat bantu untuk mengidentifikasi pasien yang berisiko tinggi.
    - **Pasien:** Memerlukan informasi tentang potensi risiko mereka agar bisa melakukan pencegahan lebih dini.
    - **Peneliti:** Ingin memahami hubungan antara faktor-faktor medis & demografi terhadap kemungkinan diabetes.
    """)
    
    st.subheader("3️⃣ Formulate Data Mining Problem")
    st.write("""
    **Permasalahan Data Mining yang ingin diselesaikan:**
    - **Tipe masalah:** Klasifikasi Supervised Learning
    - **Label target:** diabetes (1 = memiliki diabetes, 0 = tidak memiliki diabetes)
    - **Fitur yang digunakan:**
        - age (usia pasien)
        - gender (jenis kelamin)
        - BMI (indeks massa tubuh)
        - hypertension (0 = tidak, 1 = ya)
        - heart disease (0 = tidak, 1 = ya)
        - smoking history (No Info = Tidak Ada Info, Never = Tidak Pernah, Former = Mantan, Current = Saat Ini, Not Current = Tidak Saat Ini)
        - HbA1c level (rata-rata kadar gula darah dalam 2-3 bulan terakhir)
        - blood glucose level (kadar gula darah saat ini)
    """)
    
    st.subheader("4️⃣ Determine Project Objectives")
    st.write("""
    **Kriteria keberhasilan proyek:**
    - Akurasi model minimal 85% dalam memprediksi diabetes.
    - Precision dan recall minimal 80% untuk menghindari banyaknya false positives dan false negatives.
    - Dapat memberikan interpretasi klasifikasi dan rekomendasi berdasarkan sumber yang valid seperti jurnal.
    - Model yang ringan dan cepat sehingga bisa digunakan dalam sistem kesehatan berbasis web atau mobile.
    """)
    
elif selected_stage == "Data Understanding":
    st.header("Data Understanding")
    st.write("Setelah memahami permasalahan bisnis, langkah berikutnya adalah memahami data yang tersedia. Mencakup eksplorasi dataset, pemeriksaan distribusi variabel, identifikasi nilai yang hilang, serta analisis hubungan antara fitur dengan target. Dalam proyek ini, kita memiliki data seperti jenis kelamin, usia, riwayat hipertensi, riwayat penyakit jantung, riwayat merokok, BMI, kadar HbA1c, dan kadar glukosa darah, yang akan dianalisis untuk menemukan pola yang relevan dengan diabetes.")
    
    # Menampilkan 10 Data Pertama & Total Data
    st.subheader("Dataframe")
    st.dataframe(df.head(10))
    st.write(f"**Total Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

    # Mengecek Missing Value
    st.subheader("Missing Values per Kolom")
    missing_values = df.isna().sum().reset_index()
    missing_values.columns = ["Kolom", "Jumlah"]
    missing_values["Jumlah"] = missing_values["Jumlah"].astype(str)
    st.write(missing_values)

    # Menghitung Data Duplicate
    duplicates = df.duplicated().sum()
    st.subheader("Data Duplicates")
    st.write(f"**Total Data Duplicate:** {duplicates}")

    # Mengecek Outlier dengan Boxplot
    st.subheader("Outlier")
    st.write("**Outlier Check (Boxplot BMI, HbA1c, Blood Glucose):**")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(y=df["bmi"], ax=ax[0])
    ax[0].set_title("BMI")
    sns.boxplot(y=df["HbA1c_level"], ax=ax[1])
    ax[1].set_title("HbA1c Level")
    sns.boxplot(y=df["blood_glucose_level"], ax=ax[2])
    ax[2].set_title("Blood Glucose Level")
    st.pyplot(fig)
    
    st.subheader("Interpretasi Boxplot")

    st.write("""
    Dari visualisasi boxplot pada variabel **BMI, HbA1c Level, dan Blood Glucose Level**, ditemukan beberapa outlier yang dapat diinterpretasikan sebagai berikut:

    - **BMI (Body Mass Index)**  
    Terdapat banyak outlier di bagian atas sebanyak **7086** yang menunjukkan individu dengan indeks massa tubuh yang sangat tinggi. Ini mengindikasikan adanya kasus obesitas ekstrem dalam dataset.

    - **HbA1c Level**  
    Beberapa outlier ditemukan pada kadar **HbA1c** yang tinggi sebanyak **1315**. Ini bisa menjadi tanda bahwa ada individu dengan kondisi diabetes atau kontrol gula darah yang buruk.

    - **Blood Glucose Level**  
    Outlier di bagian atas sebanyak **2038** menunjukkan individu dengan kadar gula darah yang sangat tinggi (di atas 250 mg/dL). Ini bisa mengindikasikan adanya kasus hiperglikemia atau diabetes yang tidak terkontrol.
    """)

    # Mengecek Tipe Data
    st.subheader("Tipe Data")
    df_types = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.values
    })
    st.write(df_types)

    # Visualisasi Distribusi Variabel yang Mempengaruhi Diabetes
    st.subheader("Distribusi Variabel yang Mempengaruhi Diabetes")
    
    diabetes_mapping = {0: "Negatif Diabetes", 1: "Positif Diabetes"}
    hypertension = {0: "Tidak", 1: "Ya"}
    heart_disease_mapping = {0: "Tidak", 1: "Ya"}

    df["diabetes"] = df["diabetes"].map(diabetes_mapping)
    df["hypertension"] = df["hypertension"].map(hypertension)
    df["heart_disease"] = df["heart_disease"].map(heart_disease_mapping)
    
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Distribusi Variabel yang Mempengaruhi Diabetes", fontsize=16, fontweight='bold')

    # Distribusi Usia
    sns.histplot(df, x="age", hue="diabetes", bins=30, kde=True, element="step", ax=axes[0, 0])
    axes[0, 0].set_title("Distribusi Usia (Age)")

    # Distribusi BMI
    sns.histplot(df, x="bmi", hue="diabetes", bins=30, kde=True, element="step", ax=axes[0, 1])
    axes[0, 1].set_title("Distribusi BMI (Body Mass Index)")

    # Distribusi HbA1c Level
    sns.histplot(df, x="HbA1c_level", hue="diabetes", bins=30, kde=True, element="step", ax=axes[1, 0])
    axes[1, 0].set_title("Distribusi HbA1c Level")

    # Distribusi Blood Glucose Level
    sns.histplot(df, x="blood_glucose_level", hue="diabetes", bins=30, kde=True, element="step", ax=axes[1, 1])
    axes[1, 1].set_title("Distribusi Blood Glucose Level")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
    
    st.subheader("Interpretasi Distribusi Variabel")

    st.write("""
    Visualisasi di atas menunjukkan distribusi beberapa variabel penting dalam dataset terkait diabetes, dengan kategori **Negatif Diabetes** (biru) dan **Positif Diabetes** (oranye). Berikut adalah analisisnya:

    - **Distribusi Usia (Age)**  
    Mayoritas pasien dengan diabetes ditemukan pada kelompok usia di atas 50 tahun. Menunjukkan bahwa risiko diabetes meningkat seiring bertambahnya usia.

    - **Distribusi BMI (Body Mass Index)**  
    Distribusi BMI memiliki puncak di sekitar 25, yang menandakan bahwa sebagian besar individu dalam dataset memiliki indeks massa tubuh dalam rentang normal atau overweight. Namun, individu dengan **BMI lebih tinggi** lebih cenderung memiliki diabetes.

    - **Distribusi HbA1c Level**  
    Mayoritas individu dengan diabetes memiliki **HbA1c Level di atas 6.5**, yang sesuai dengan kriteria medis untuk diagnosis diabetes. Menunjukkan bahwa kadar **HbA1c dapat menjadi indikator kuat untuk prediksi diabetes**.

    - **Distribusi Blood Glucose Level**  
    Individu dengan kadar gula darah lebih dari **150 mg/dL** lebih sering ditemukan dalam kategori diabetes. **Puncak distribusi pada nilai tertentu** menunjukkan bahwa banyak individu dalam dataset memiliki kadar gula darah yang khas untuk penderita diabetes.

    """)

    # Visualisasi Hypertension vs Diabetes
    st.subheader("Hypertension vs Diabetes")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax_hypertension = sns.countplot(data=df, x="hypertension", hue="diabetes")
    plt.title("Hypertension vs Diabetes")

    for p in ax_hypertension.patches:
        if p.get_height() > 0: 
            ax_hypertension.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2), textcoords='offset points')
            
    st.pyplot(fig)
    
    st.subheader("Interpretasi Hubungan Hipertensi dan Diabetes")

    st.write("""
    Grafik di atas menunjukkan distribusi individu dengan dan tanpa hipertensi serta hubungannya dengan diabetes. 

    - Mayoritas individu dalam dataset **tidak memiliki hipertensi** (bar biru lebih tinggi di kategori "Tidak"), dan sebagian besar dari mereka juga **tidak menderita diabetes**.
    - Namun, dari individu yang tidak memiliki hipertensi, terdapat **6.412 orang yang menderita diabetes**, menunjukkan bahwa diabetes bisa terjadi meskipun tanpa hipertensi.
    - Di sisi lain, pada kelompok individu **yang memiliki hipertensi**, terdapat **2.088 orang yang menderita diabetes**. Jumlah ini memang lebih kecil dibandingkan kategori tanpa hipertensi, tetapi **rasio penderita diabetes terhadap total populasi lebih tinggi di kelompok hipertensi**.
    - Menunjukkan bahwa meskipun hipertensi bukan satu-satunya faktor risiko, individu dengan hipertensi **lebih cenderung memiliki diabetes dibandingkan individu tanpa hipertensi**.

    """)

    # 8️⃣ Visualisasi Heart Disease vs Diabetes
    st.subheader("Heart Disease vs Diabetes")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax_heart_disease = sns.countplot(data=df, x="heart_disease", hue="diabetes")
    plt.title("Heart Disease vs Diabetes")

    for p in ax_heart_disease.patches:
        if p.get_height() > 0: 
            ax_heart_disease.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2), textcoords='offset points')
    st.pyplot(fig)
    
    st.subheader("Hubungan Penyakit Jantung dan Diabetes")

    st.write("""
    Grafik di atas menunjukkan hubungan antara penyakit jantung dan diabetes.

    - Sebagian besar individu dalam dataset **tidak memiliki penyakit jantung** (bar biru tinggi pada kategori "Tidak"), dan mayoritas dari mereka **tidak menderita diabetes** (88.825 orang).
    - Namun, dari kelompok yang **tidak memiliki penyakit jantung**, ada **7.233 orang yang menderita diabetes**. Ini menunjukkan bahwa diabetes tetap bisa terjadi meskipun tanpa adanya riwayat penyakit jantung.
    - Pada individu yang **memiliki penyakit jantung**, terdapat **1.267 orang yang juga memiliki diabetes**, sedangkan **2.675 orang tidak memiliki diabetes**.
    - **Persentase penderita diabetes lebih tinggi pada kelompok dengan penyakit jantung** dibandingkan kelompok tanpa penyakit jantung, yang mengindikasikan adanya korelasi antara kedua kondisi ini.

    """)
    
elif selected_stage == "Data Preparation":
    st.header("Data Preparation")
    st.write("Tahap ini melibatkan pemrosesan data sebelum dimasukkan ke dalam model Machine Learning. Data yang kotor atau tidak terstruktur akan dibersihkan, termasuk menangani nilai yang hilang, encoding variabel kategori, membersihkan outlier, dan melakukan oversampling agar data target seimbang serta membagi data menjadi set pelatihan dan pengujian. Misalnya, pada proyek prediksi diabetes, variabel seperti gender dan smoking history perlu diubah ke dalam format numerik agar dapat diproses oleh algoritma Machine Learning.")
    
     # **🔹 1. Remove Duplicates**
    st.subheader("1️⃣ Pembersihan Data - Remove Duplicates")
    df, before_dedup, after_dedup = remove_duplicates(df)
    st.write(f"Jumlah data sebelum remove duplicate: **{before_dedup}**")
    st.write(f"Jumlah data setelah remove duplicate: **{after_dedup}**")

    # **🔹 2. Remove Outliers**
    st.subheader("2️⃣ Pembersihan Data - Remove Outliers")
    numerical_columns = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    df_cleaned = remove_outliers_iqr(df, numerical_columns)
    st.write(f"Jumlah data setelah menghapus outlier: **{len(df_cleaned)}**")

    # **🔹 3. Encoding Categorical Variables**
    st.subheader("3️⃣ Encode Categorical Variables")
    categorical_columns = ['gender', 'smoking_history', 'hypertension', 'heart_disease', 'diabetes']
    df_encoded, label_encoders = encode_categorical(df_cleaned, categorical_columns)
    st.write("**Data setelah encoding kategori:**")
    st.write(df_encoded.head(10))

    # **🔹 4. Membagi Features dan Target**
    st.subheader("4️⃣ Membagi Features dan Target")
    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']
    st.write(f"Jumlah fitur: {X.shape[1]}")
    st.write(f"Jumlah sampel: {X.shape[0]}")

    # **🔹 5. Oversampling dengan SMOTE**
    st.subheader("5️⃣ Melakukan Oversampling (SMOTE)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = ['0 (Negatif)', '1 (Positif)']

    before_counts = dict(Counter(y))

    axes[0].bar(labels, before_counts.values(), color=['blue', 'red'])
    axes[0].set_title("Sebelum SMOTE")
    axes[0].set_xlabel("Kelas")
    axes[0].set_ylabel("Jumlah Sampel")

    for i, v in enumerate(before_counts.values()):
        axes[0].text(i, v + 1, str(v), ha='center', fontsize=12)
        
    X_resampled, y_resampled = oversample_smote(X, y)

    after_counts = dict(Counter(y_resampled))
    
    axes[1].bar(labels, after_counts.values(), color=['blue', 'red'])
    axes[1].set_title("Setelah SMOTE")
    axes[1].set_xlabel("Kelas")
    axes[1].set_ylabel("Jumlah Sampel")

    for i, v in enumerate(after_counts.values()):
        axes[1].text(i, v + 1, str(v), ha='center', fontsize=12)

    st.pyplot(fig)

    # **🔹 6. Train-Test Split**
    st.subheader("6️⃣ Membagi Dataset - Train Test Split (80:20)")
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    st.write(f"Data Train: {X_train.shape[0]} samples")
    st.write(f"Data Test: {X_test.shape[0]} samples")

elif selected_stage == "Modeling":
    st.header("Modeling")
    st.write("Setelah data siap, kami mulai membangun model Machine Learning. Berbagai algoritma seperti K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) dapat digunakan untuk membuat prediksi. Model akan dilatih menggunakan data yang telah diproses, dan parameter model akan dioptimalkan untuk meningkatkan kinerjanya. Pada tahap ini, dilakukan eksperimen dengan berbagai model dan hyperparameter tuning untuk mendapatkan hasil terbaik.")
    
    # Penjelasan Inisialisasi Model
    st.subheader("Inisialisasi Model")
    st.write(
        """
        Pada tahap ini, kami akan menginisialisasi dua model Machine Learning:
        - **K-Nearest Neighbors (KNN)** dengan jumlah tetangga `n_neighbors=5`
        - **Support Vector Classifier (SVC)** dengan hyperparameter `C=1.0`
        """
    )
    
    # Menampilkan kode inisialisasi model
    st.code(
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC

        # Inisialisasi Model
        knn_model = KNeighborsClassifier(n_neighbors=5)
        svc_model = SVC(C=1.0, random_state=42)
                """,
        language="python",
    )

    # Penjelasan Training Model
    st.subheader("Training Model")
    st.write(
        """
        Setelah model diinisialisasi, langkah selanjutnya adalah melakukan **training (pelatihan model)** 
        menggunakan dataset training (`X_train, y_train`) dan melakukan prediksi pada dataset uji (`X_test`).
        """
    )

    # Menampilkan kode Training Model
    st.code(
        """
        # Training Model
        y_pred_knn = knn_model.fit(X_train, y_train).predict(X_test)
        y_pred_svc = svc_model.fit(X_train, y_train).predict(X_test)
                """,
        language="python",
    )
    
elif selected_stage == "Evaluation":
    st.header("Evaluation")
    st.write("Model yang telah dibangun perlu dievaluasi untuk memastikan performanya sesuai dengan tujuan bisnis. Metrik seperti akurasi, precision, recall, F1-score, dan AUC-ROC curve serta Confusion matrix kami gunakan untuk mengukur seberapa baik model dalam melakukan klasifikasi. Jika hasil evaluasi belum memuaskan, maka model dapat di-tuning ulang atau dilakukan pemrosesan data lebih lanjut agar prediksi lebih akurat.")
    
    X_test = joblib.load("./utils/X_test.pkl")
    y_test = joblib.load("./utils/y_test.pkl")

    y_pred_knn = model_knn.predict(X_test)
    y_pred_svm = model_svm.predict(X_test)

    # ** Akurasi Model**
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    st.write("### 1️⃣ Akurasi Model")
    st.write(f"**Accuracy KNN**: {accuracy_knn:.4f}")
    st.write(f"**Accuracy SVM**: {accuracy_svm:.4f}")

    # ** Classification Report**
    st.write("### 2️⃣ Classification Report")
    st.write("Classification Report - KNN")
    report_knn = classification_report(y_test, y_pred_knn, target_names=label_encoders["diabetes"].classes_, output_dict=True)
    report_knn_df = pd.DataFrame(report_knn).transpose()
    st.dataframe(report_knn_df)
    
    st.write("Classification Report - SVM")
    report_svm = classification_report(y_test, y_pred_svm, target_names=label_encoders["diabetes"].classes_, output_dict=True)
    report_svm_df = pd.DataFrame(report_svm).transpose()
    st.dataframe(report_svm_df)

    # ** ROC & AUC Curve**
    st.write("### 3️⃣ ROC & AUC Curve")
    
    n_classes = len(set(y_test))
    if n_classes > 2:
        y_test_bin = LabelBinarizer(y_test, classes=list(set(y_test)))
    else:
        y_test_bin = y_test

    y_score_knn = model_knn.predict_proba(X_test)[:, 1]
    y_score_svc = model_svm.decision_function(X_test)

    fpr_knn, tpr_knn, _ = roc_curve(y_test_bin, y_score_knn)
    fpr_svc, tpr_svc, _ = roc_curve(y_test_bin, y_score_svc)

    auc_knn = auc(fpr_knn, tpr_knn)
    auc_svc = auc(fpr_svc, tpr_svc)

    # Visualisasi ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', linestyle='--')
    plt.plot(fpr_svc, tpr_svc, label=f'SVM (AUC = {auc_svc:.2f})', linestyle='-.')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='dotted')

    # Pengaturan plot
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    st.pyplot(plt)
    
    st.subheader("Interpretasi ROC Curve")
    st.write("""
    Kurva ROC digunakan untuk mengevaluasi kinerja model klasifikasi berdasarkan trade-off antara **True Positive Rate (TPR)** dan **False Positive Rate (FPR)**.

    - **Model KNN (AUC = 0.97)** memiliki performa yang lebih baik dibandingkan **SVM (AUC = 0.95)** karena nilai **AUC lebih tinggi**.
    - Semakin mendekati nilai **AUC = 1**, semakin baik model dalam membedakan antara kelas positif dan negatif.
    - Kurva KNN lebih tinggi dibandingkan kurva SVM, menunjukkan bahwa KNN lebih akurat dalam memprediksi positif dengan lebih sedikit kesalahan.
    """)
    
    # ** Confusion Matrix **
    st.write("### 4️⃣ Confusion Matrix")
    
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    def plot_confusion_matrix(cm, model_name):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig) 

    st.write("#### Confusion Matrix - KNN")
    plot_confusion_matrix(cm_knn, "KNN")
    
    st.subheader("Interpretasi Confusion Matrix - KNN")
    st.write("""
    - **True Negatives (TN)**: 14,539 -> Model memprediksi negatif dengan benar.
    - **False Positives (FP)**: 2,025 -> Model salah memprediksi sebagai positif padahal negatif.
    - **False Negatives (FN)**: 213 -> Model salah memprediksi sebagai negatif padahal positif.
    - **True Positives (TP)**: 16,654 -> Model memprediksi positif dengan benar.

    **Akurasi KNN** relatif tinggi karena jumlah TP dan TN lebih dominan dibandingkan FP dan FN.
    """)

    st.write("#### Confusion Matrix - SVM")
    plot_confusion_matrix(cm_svm, "SVM")
    st.subheader("Confusion Matrix - SVM")
    st.write("""
    - **True Negatives (TN)**: 13,572 -> Model memprediksi negatif dengan benar.
    - **False Positives (FP)**: 2,992 -> Model salah memprediksi sebagai positif padahal negatif.
    - **False Negatives (FN)**: 1,046 -> Model salah memprediksi sebagai negatif padahal positif.
    - **True Positives (TP)**: 15,821 -> Model memprediksi positif dengan benar.

    **Akurasi SVM** masih tinggi, tetapi jumlah FP dan FN lebih besar dibandingkan KNN, menunjukkan bahwa KNN sedikit lebih unggul dalam memprediksi dengan benar.
    """)
    
elif selected_stage == "Deployment":
    st.title("Deployment")
    st.write("Tahap terakhir adalah menerapkan model ke dalam lingkungan produksi agar dapat digunakan oleh pengguna akhir. Model yang sudah dilatih dan dievaluasi akan dikemas dalam bentuk API atau aplikasi berbasis web menggunakan Streamlit. API akan menerima input dari pengguna, memproses data, menjalankan model prediksi, dan mengembalikan hasil prediksi secara real-time. Dengan deployment ini, pengguna dapat dengan mudah mengakses layanan prediksi diabetes tanpa harus memiliki pengetahuan teknis tentang Machine Learning.")

    # **Form Input Data**
    st.header("Masukkan Data")
    st.write("""
    Masukkan data kesehatan Anda untuk mendapatkan prediksi apakah Anda berisiko mengalami diabetes atau tidak.
    """)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
        age = st.number_input("Usia (tahun)", min_value=1, max_value=80, value=1)
        hypertension = st.selectbox("Riwayat Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    with col2:
        smoking_options = ["No Info", "never", "former", "current", "not current"]
        smoking_options_capitalized = [option.title() for option in smoking_options]    
        smoking_history = st.selectbox("Riwayat Merokok", smoking_options_capitalized)
        smoking_history_original = smoking_options[smoking_options_capitalized.index(smoking_history)]
        
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=95.7, value=10.0, step=0.1)
        hba1c_level = st.number_input("Kadar HbA1c (%)", min_value=3.5, max_value=9.0, value=3.5, step=0.1)
        blood_glucose_level = st.number_input("Kadar Glukosa Darah (mg/dL)", min_value=80, max_value=300, value=80)

    # **Konversi Input ke Format Model**
    input_data = {
        "gender": label_encoders["gender"].transform([gender])[0],
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": label_encoders["smoking_history"].transform([smoking_history_original])[0],
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level,
    }

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # **Prediksi**
    if st.button("Prediksi"):
        prediction = model_knn.predict(input_array)
        result = "🟥 **Positif Diabetes**" if prediction[0] == 1 else "🟩 **Negatif Diabetes**"

        # Interpretasi & Rekomendasi
        if prediction[0] == 1:
            interpretation = "⚠️ **Model memprediksi bahwa Anda memiliki kemungkinan mengalami diabetes.**"
            follow_up = """
            **Rekomendasi:**
            - **Konsultasi dengan Dokter** untuk pemeriksaan lebih lanjut.
            - **Pantau Pola Makan** dengan mengurangi gula dan meningkatkan konsumsi serat.
            - **Olahraga Rutin** seperti jalan cepat, jogging, atau bersepeda.
            - **Cek Kesehatan Rutin** untuk memantau kadar HbA1c dan glukosa darah.
            """
        else:
            interpretation = "✅ **Model memprediksi bahwa Anda tidak memiliki diabetes.**"
            follow_up = """
            **Rekomendasi:**
            - **Tetap Jaga Pola Hidup Sehat** dengan konsumsi makanan seimbang dan olahraga rutin.
            - **Cek Kesehatan Secara Berkala** untuk memastikan kadar gula darah tetap normal.
            - **Kelola Stres** agar kesehatan tetap optimal.
            """

        # **Tampilkan Hasil**
        st.subheader("Hasil Prediksi")
        st.write(result)
        st.write(interpretation)
        st.info(follow_up)
    
    # **Dokumentasi API**
    st.header("Dokumentasi API")

    st.write("""
    API ini memungkinkan pengguna untuk mengirimkan data kesehatan dan mendapatkan prediksi risiko diabetes berdasarkan model Machine Learning.

    ### **Endpoint Prediksi**
    **URL:** `/predict`  
    **Method:** `POST`  
    **Content-Type:** `application/json`

    #### **Contoh Request Body**
    """)

    st.code("""
    {
        "gender": "Female",
        "age": 53,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "No Info",
        "bmi": 31.75,
        "HbA1c_level": 4.0,
        "blood_glucose_level": 155
    }
    """, language="json")

    st.write("""
    #### **Contoh Response**
    """)

    st.code("""
    {
        "prediction": "Positive Diabetes",
        "interpretation": "Model memprediksi bahwa Anda memiliki kemungkinan mengalami diabetes.",
        "recommendations": [
            "Konsultasi dengan dokter untuk pemeriksaan lebih lanjut.",
            "Pantau pola makan dengan mengurangi gula dan meningkatkan konsumsi serat.",
            "Olahraga rutin seperti jalan cepat, jogging, atau bersepeda.",
            "Cek kesehatan rutin untuk memantau kadar HbA1c dan glukosa darah."
        ]
    }
    """, language="json")