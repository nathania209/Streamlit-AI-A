# =============================================================
# 📦 IMPORT LIBRARY
# =============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset dari sklearn
from sklearn import datasets

# Import model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Import model klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

# Import utilitas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# =============================================================
# ⚙️ KONFIGURASI HALAMAN
# =============================================================
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide",  # Layout lebar sesuai permintaan
    initial_sidebar_state="expanded"
)

# Styling CSS kustom agar UI terlihat cantik
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6rem;
            border: none;
        }
        .stButton > button:hover {
            opacity: 0.9;
            color: white;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #764ba2;
        }
    </style>
""", unsafe_allow_html=True)


# =============================================================
# 📚 FUNGSI LOAD DATASET (dengan caching agar cepat)
# =============================================================
@st.cache_data
def load_dataset(name: str):
    """Memuat dataset toy dari sklearn berdasarkan nama."""
    if name == "Diabetes":
        data = datasets.load_diabetes()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = None
    elif name == "California Housing":
        data = datasets.fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = None
    elif name == "Iris":
        data = datasets.load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = list(data.target_names)
    elif name == "Wine":
        data = datasets.load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = list(data.target_names)
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = list(data.target_names)
    elif name == "Digits":
        data = datasets.load_digits()
        X, y = data.data, data.target
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
    else:
        raise ValueError(f"Dataset {name} tidak dikenal.")
    
    return X, y, feature_names, target_names


# =============================================================
# 🏷️ DAFTAR DATASET DAN MODEL
# =============================================================
REGRESSION_DATASETS = ["Diabetes", "California Housing"]
CLASSIFICATION_DATASETS = ["Iris", "Wine", "Breast Cancer", "Digits"]

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
}


# =============================================================
# 🎛️ SIDEBAR - PANEL KONTROL
# =============================================================
with st.sidebar:
    st.markdown("## 🎛️ Panel Kontrol")
    st.markdown("---")

    # Pilih tipe task
    task_type = st.radio(
        "🧠 Pilih Tipe Task:",
        ["Regresi", "Klasifikasi"],
        help="Regresi untuk target kontinu, Klasifikasi untuk target kategorikal"
    )

    st.markdown("---")

    # Pilih dataset sesuai task
    if task_type == "Regresi":
        dataset_name = st.selectbox("📊 Pilih Dataset:", REGRESSION_DATASETS)
        model_name = st.selectbox("🤖 Pilih Model:", list(REGRESSION_MODELS.keys()))
    else:
        dataset_name = st.selectbox("📊 Pilih Dataset:", CLASSIFICATION_DATASETS)
        model_name = st.selectbox("🤖 Pilih Model:", list(CLASSIFICATION_MODELS.keys()))

    st.markdown("---")

    # Slider test size (10% - 50%)
    test_size = st.slider(
        "📏 Test Size (%)",
        min_value=10, max_value=50, value=20, step=5,
        help="Proporsi data untuk testing"
    ) / 100.0

    # Opsi tambahan: scaling
    use_scaling = st.checkbox(
        "⚖️ Standard Scaling",
        value=True,
        help="Terapkan StandardScaler pada fitur (disarankan untuk KNN, SVM, dll.)"
    )

    st.markdown("---")

    # Tombol train
    train_button = st.button("🚀 Train Model", use_container_width=True)

    st.markdown("---")
    st.markdown("💡 **Tips:** Pilih dataset, model, lalu tekan **Train Model**.")


# =============================================================
# 🏠 HEADER UTAMA
# =============================================================
st.markdown('<div class="main-header">🤖 ML Playground Interaktif</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Task: <b>{task_type}</b> | Dataset: <b>{dataset_name}</b> | Model: <b>{model_name}</b></div>',
    unsafe_allow_html=True
)


# =============================================================
# 🚂 PROSES TRAINING (disimpan di session_state)
# =============================================================
if train_button:
    try:
        # Load dataset
        with st.spinner("📥 Memuat dataset..."):
            X, y, feature_names, target_names = load_dataset(dataset_name)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task_type == "Klasifikasi" else None
        )

        # Scaling (opsional)
        if use_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Inisialisasi model
        if task_type == "Regresi":
            model = REGRESSION_MODELS[model_name]()
        else:
            # SVM butuh probability=True untuk beberapa fungsi
            if model_name == "SVM":
                model = CLASSIFICATION_MODELS[model_name](probability=True)
            else:
                model = CLASSIFICATION_MODELS[model_name]()

        # Training
        with st.spinner(f"🏋️ Melatih model {model_name}..."):
            model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Simpan ke session_state
        st.session_state["trained"] = True
        st.session_state["task_type"] = task_type
        st.session_state["dataset_name"] = dataset_name
        st.session_state["model_name"] = model_name
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["target_names"] = target_names
        st.session_state["X_test_shape"] = X_test.shape
        st.session_state["X_train_shape"] = X_train.shape

        st.success(f"✅ Model **{model_name}** berhasil dilatih!")

    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")


# =============================================================
# 📊 TAMPILKAN HASIL JIKA MODEL SUDAH DILATIH
# =============================================================
if st.session_state.get("trained", False):
    # Ambil data dari session_state
    task = st.session_state["task_type"]
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    target_names = st.session_state.get("target_names", None)

    # Info ringkas
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("📚 Data Train", f"{st.session_state['X_train_shape'][0]} sampel")
    with col_info2:
        st.metric("🧪 Data Test", f"{st.session_state['X_test_shape'][0]} sampel")
    with col_info3:
        st.metric("🔢 Jumlah Fitur", f"{st.session_state['X_test_shape'][1]} fitur")

    st.markdown("---")

    # Buat tab
    tab1, tab2 = st.tabs(["📈 Visualisasi", "📊 Evaluation Metrics"])

    # ========================================================
    # 📈 TAB 1: VISUALISASI
    # ========================================================
    with tab1:
        if task == "Regresi":
            st.subheader("📈 Actual vs Predicted")

            # Scatter plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, color="#667eea",
                       edgecolors="white", s=60, label="Prediksi")

            # Garis diagonal ideal (y = x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    "r--", lw=2, label="Ideal (y = x)")

            ax.set_xlabel("Nilai Aktual", fontsize=12)
            ax.set_ylabel("Nilai Prediksi", fontsize=12)
            ax.set_title(f"Actual vs Predicted — {st.session_state['model_name']}",
                         fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            sns.despine()
            st.pyplot(fig)

            st.info("💡 **Interpretasi:** Semakin dekat titik-titik ke garis merah putus-putus, semakin akurat prediksi model.")

        else:  # Klasifikasi
            st.subheader("🔲 Confusion Matrix")

            # Hitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            labels = target_names if target_names else sorted(np.unique(y_test))

            # Plot heatmap confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels,
                cbar=True, square=True, ax=ax,
                annot_kws={"size": 12, "weight": "bold"}
            )
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(f"Confusion Matrix — {st.session_state['model_name']}",
                         fontsize=14, fontweight="bold")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)

            st.info("💡 **Interpretasi:** Diagonal utama = prediksi benar. Angka di luar diagonal = kesalahan klasifikasi.")

    # ========================================================
    # 📊 TAB 2: EVALUATION METRICS
    # ========================================================
    with tab2:
        if task == "Regresi":
            st.subheader("📊 Metrik Evaluasi Regresi")

            # Hitung metrik
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Tampilkan sebagai metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📏 MAE", f"{mae:.4f}",
                          help="Mean Absolute Error — rata-rata error absolut")
            with c2:
                st.metric("📐 MSE", f"{mse:.4f}",
                          help="Mean Squared Error — rata-rata error kuadrat")
            with c3:
                st.metric("📊 RMSE", f"{rmse:.4f}",
                          help="Root Mean Squared Error — akar dari MSE")
            with c4:
                st.metric("🎯 R² Score", f"{r2:.4f}",
                          help="Koefisien Determinasi — semakin dekat 1 semakin baik")

            st.markdown("---")

            # Tabel ringkasan
            st.markdown("### 📋 Tabel Ringkasan")
            df_metrics = pd.DataFrame({
                "Metrik": ["MAE", "MSE", "RMSE", "R² Score"],
                "Nilai": [mae, mse, rmse, r2],
                "Deskripsi": [
                    "Error absolut rata-rata (semakin kecil semakin baik)",
                    "Error kuadrat rata-rata (semakin kecil semakin baik)",
                    "Akar MSE (semakin kecil semakin baik)",
                    "Proporsi variansi yang dijelaskan (semakin dekat 1 semakin baik)"
                ]
            })
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Bar chart perbandingan nilai aktual vs prediksi (10 sampel pertama)
            st.markdown("### 🔍 Perbandingan Aktual vs Prediksi (10 sampel pertama)")
            df_compare = pd.DataFrame({
                "Aktual": y_test[:10],
                "Prediksi": y_pred[:10]
            }).reset_index().rename(columns={"index": "Sampel"})
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

        else:  # Klasifikasi
            st.subheader("📊 Metrik Evaluasi Klasifikasi")

            # Tentukan average berdasarkan jumlah kelas
            n_classes = len(np.unique(y_test))
            avg = "binary" if n_classes == 2 else "weighted"

            # Hitung metrik
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

            # Tampilkan metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("✅ Accuracy", f"{acc:.4f}",
                          help="Proporsi prediksi benar")
            with c2:
                st.metric("🎯 Precision", f"{prec:.4f}",
                          help=f"Precision ({avg} average)")
            with c3:
                st.metric("🔍 Recall", f"{rec:.4f}",
                          help=f"Recall ({avg} average)")
            with c4:
                st.metric("⚖️ F1-Score", f"{f1:.4f}",
                          help=f"F1-Score ({avg} average)")

            st.markdown("---")

            # Classification report
            st.markdown("### 📄 Classification Report")
            report = classification_report(
                y_test, y_pred,
                target_names=[str(t) for t in target_names] if target_names else None,
                output_dict=True, zero_division=0
            )
            df_report = pd.DataFrame(report).transpose()
            # Format angka
            st.dataframe(
                df_report.style.format("{:.4f}").background_gradient(cmap="Purples", axis=0),
                use_container_width=True
            )

else:
    # Tampilan awal sebelum training
    st.info("👈 **Silakan pilih konfigurasi di sidebar, lalu tekan tombol 🚀 Train Model untuk memulai.**")

    # Tampilkan preview dataset
    st.markdown("### 📊 Preview Dataset")
    try:
        X_prev, y_prev, feat_names_prev, _ = load_dataset(dataset_name)
        df_preview = pd.DataFrame(X_prev, columns=feat_names_prev)
        df_preview["🎯 Target"] = y_prev
        st.dataframe(df_preview.head(10), use_container_width=True)
        st.caption(f"Shape: {df_preview.shape[0]} baris × {df_preview.shape[1]} kolom")
    except Exception as e:
        st.warning(f"Tidak dapat memuat preview: {e}")


# =============================================================
# 📌 FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🤖 Dibuat dengan Streamlit + Scikit-Learn | ML Playground © 2024"
    "</div>",
    unsafe_allow_html=True
)