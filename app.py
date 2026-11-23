import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import certifi

from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_iris,
    load_breast_cancer,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Impor model Regresi, Klasifikasi, & Clustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(
    page_title="TAS Streamlit: Machine Learning & MongoDB",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 TES AKHIR SEMESTER: MACHINE LEARNING & MONGODB")
st.markdown("Aplikasi Web Streamlit: **Regresi, Klasifikasi, Clustering, MongoDB Retrain & Komentar Publik**")
st.markdown("---")


# --- KONEKSI MONGODB DENGAN CACHE ---
@st.cache_resource
def init_mongo_client():
    uri = st.secrets["MONGODB_URI"]
    return MongoClient(uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)


try:
    client = init_mongo_client()
    db = client["test_db"]
    iris_col = db["iris_db"]
    comments_col = db["public_comments"]
    mongo_connected = True
except Exception as e:
    mongo_connected = False
    mongo_error = str(e)


# --- FUNGSI MEMUAT DATASET ---
@st.cache_data
def load_data(name):
    if name == "Diabetes":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target
    elif name == "Iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "Wine":
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    return X, y


@st.cache_data(ttl=5)
def get_mongo_iris_data():
    if not mongo_connected:
        return []
    return list(iris_col.find())


@st.cache_data(ttl=5)
def get_mongo_comments():
    if not mongo_connected:
        return []
    return list(comments_col.find({}, {"_id": 0, "nama": 1, "komentar": 1, "timestamp": 1}).sort("_id", -1))


# --- SIDEBAR: PENGATURAN MODEL ---
st.sidebar.header("⚙️ Pengaturan Model")
task_type = st.sidebar.radio("Pilih Jenis Task", ["Regresi", "Klasifikasi", "Clustering"])

if task_type == "Regresi":
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Diabetes", "California Housing"])
    model_name = st.sidebar.selectbox(
        "Pilih Model",
        ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],
    )
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
elif task_type == "Klasifikasi":
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Iris", "Breast Cancer", "Wine"])
    model_name = st.sidebar.selectbox(
        "Pilih Model",
        [
            "Logistic Regression",
            "K-Nearest Neighbors (KNN)",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Support Vector Machine (SVM)",
        ],
    )
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
else:
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Iris", "Wine"])
    model_name = "K-Means Clustering"
    n_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

train_button = st.sidebar.button("🚀 Train Model / Run Clustering", use_container_width=True)

X, y = load_data(dataset_name)

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button:
    if task_type != "Clustering":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        if task_type == "Regresi":
            model_map = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=int(random_state)),
                "Random Forest Regressor": RandomForestRegressor(random_state=int(random_state)),
            }
        else:
            model_map = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=int(random_state)),
                "Random Forest Classifier": RandomForestClassifier(random_state=int(random_state)),
                "Support Vector Machine (SVM)": SVC(),
            }

        model = model_map[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
    else:
        model = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
        clusters = model.fit_predict(X)
        st.session_state.clusters = clusters
        st.session_state.inertia = float(model.inertia_)

    st.session_state.trained = True
    st.session_state.task_type = task_type
    st.session_state.model_name = model_name
    st.session_state.dataset_name = dataset_name
    st.session_state.X_sample = X.head(10)
    st.success(f"✅ Berhasil menjalankan **{model_name}** pada dataset **{dataset_name}**!")


# --- TABS UTAMA ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Training & Visualisasi",
    "📈 Evaluation Metrics",
    "💾 Database & Retrain",
    "💬 Komentar Publik",
])

# === TAB 1: VISUALISASI ===
with tab1:
    if not st.session_state.trained:
        st.info("👈 Silakan atur parameter di sidebar lalu klik tombol **Train Model / Run Clustering**.")
    else:
        st.subheader(f"Visualisasi Hasil - {st.session_state.model_name} ({st.session_state.dataset_name})")
        with st.expander("Lihat Pratinjau Dataset (10 Baris Pertama)"):
            st.dataframe(st.session_state.X_sample, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if st.session_state.task_type == "Regresi":
            ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.7, color="#1f77b4")
            min_val = min(st.session_state.y_test.min(), st.session_state.y_pred.min())
            max_val = max(st.session_state.y_test.max(), st.session_state.y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
        elif st.session_state.task_type == "Klasifikasi":
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Label Prediksi")
            ax.set_ylabel("Label Aktual")
            ax.set_title("Confusion Matrix")
        else:
            X_curr, _ = load_data(st.session_state.dataset_name)
            scatter = ax.scatter(
                X_curr.iloc[:, 0],
                X_curr.iloc[:, 1],
                c=st.session_state.clusters,
                cmap="viridis",
                alpha=0.75,
            )
            ax.set_xlabel(X_curr.columns[0])
            ax.set_ylabel(X_curr.columns[1])
            ax.set_title("Hasil Pengelompokan K-Means Clustering")
            plt.colorbar(scatter, ax=ax, label="Cluster")

        st.pyplot(fig)


# === TAB 2: EVALUATION METRICS ===
with tab2:
    if not st.session_state.trained:
        st.info("👈 Silakan jalankan model terlebih dahulu di sidebar.")
    else:
        st.subheader(f"Evaluation Metrics - {st.session_state.model_name}")
        if st.session_state.task_type == "Regresi":
            mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred)
            mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("MSE", f"{mse:.4f}")
            c3.metric("RMSE", f"{rmse:.4f}")
            c4.metric("R² Score", f"{r2:.4f}")

        elif st.session_state.task_type == "Klasifikasi":
            acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            prec = precision_score(st.session_state.y_test, st.session_state.y_pred, average="weighted", zero_division=0)
            rec = recall_score(st.session_state.y_test, st.session_state.y_pred, average="weighted", zero_division=0)
            f1 = f1_score(st.session_state.y_test, st.session_state.y_pred, average="weighted", zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1-Score", f"{f1:.4f}")

            st.markdown("#### 📋 Classification Report")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        else:
            st.metric("Inertia (Within-Cluster Sum of Squares)", f"{st.session_state.inertia:.4f}")
            unique, counts = np.unique(st.session_state.clusters, return_counts=True)
            cluster_df = pd.DataFrame({"Cluster": unique, "Jumlah Data": counts})
            st.dataframe(cluster_df, use_container_width=True)


# === TAB 3: MONGODB CRUD & RETRAIN IRIS ===
with tab3:
    st.subheader("💾 MongoDB CRUD & Retrain Model (Dataset Iris)")
    if not mongo_connected:
        st.error(f"❌ Gagal terhubung ke MongoDB Atlas: {mongo_error}")
    else:
        docs = get_mongo_iris_data()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Dokumen di MongoDB", len(docs))
        m2.metric("Database", "test_db")
        m3.metric("Collection", "iris_db")

        st.markdown("---")
        col_form, col_table = st.columns([1, 1.4])

        with col_form:
            st.markdown("#### ➕ Tambah Data Iris Baru")
            with st.form("form_insert_iris"):
                sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
                sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
                petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
                petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=15.0, value=0.2, step=0.1)
                species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])
                submitted_iris = st.form_submit_button("Simpan ke MongoDB", use_container_width=True)

                if submitted_iris:
                    new_doc = {
                        "sepal_length": float(sepal_length),
                        "sepal_width": float(sepal_width),
                        "petal_length": float(petal_length),
                        "petal_width": float(petal_width),
                        "species": species,
                        "timestamp": datetime.now(),
                    }
                    iris_col.insert_one(new_doc)
                    get_mongo_iris_data.clear()
                    st.success("✅ Data Iris berhasil disimpan ke MongoDB!")
                    st.rerun()

        with col_table:
            st.markdown("#### 📄 Data Tersimpan di MongoDB")
            if docs:
                df_mongo = pd.DataFrame(docs)
                df_mongo["_id"] = df_mongo["_id"].astype(str)
                st.dataframe(df_mongo, use_container_width=True)

                # Hapus berdasarkan _id
                id_options = df_mongo["_id"].tolist()
                selected_id = st.selectbox("Pilih `_id` dokumen untuk dihapus:", id_options)
                if st.button("🗑️ Hapus Dokumen Terpilih", type="secondary"):
                    iris_col.delete_one({"_id": ObjectId(selected_id)})
                    get_mongo_iris_data.clear()
                    st.warning("Dokumen berhasil dihapus.")
                    st.rerun()
            else:
                st.info("Belum ada data tambahan di collection `iris_db`.")

        st.markdown("---")
        st.markdown("#### 🔄 Retrain Model (Sklearn Iris + Data MongoDB)")
        if st.button("🚀 Retrain Sekarang", type="primary"):
            iris_sklearn = load_iris()
            X_base = pd.DataFrame(
                iris_sklearn.data,
                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            )
            y_base = pd.Series(iris_sklearn.target, name="target")

            species_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
            if docs:
                df_extra = pd.DataFrame(docs)
                df_extra = df_extra[df_extra["species"].isin(species_map.keys())]
                if not df_extra.empty:
                    X_extra = df_extra[["sepal_length", "sepal_width", "petal_length", "petal_width"]].astype(float)
                    y_extra = df_extra["species"].map(species_map)
                    X_combined = pd.concat([X_base, X_extra], ignore_index=True)
                    y_combined = pd.concat([y_base, y_extra], ignore_index=True)
                else:
                    X_combined, y_combined = X_base, y_base
            else:
                X_combined, y_combined = X_base, y_base

            X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
            rf_retrain = RandomForestClassifier(random_state=42)
            rf_retrain.fit(X_tr, y_tr)
            y_pr = rf_retrain.predict(X_te)

            acc_re = accuracy_score(y_te, y_pr)
            st.success(
                f"✅ Retraining selesai dengan total **{len(X_combined)} sampel** "
                f"(150 sklearn + {len(X_combined) - 150} MongoDB)! Akurasi Random Forest: **{acc_re:.4f}**"
            )

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**Classification Report**")
                rep_re = classification_report(y_te, y_pr, output_dict=True)
                st.dataframe(pd.DataFrame(rep_re).transpose(), use_container_width=True)
            with col_r2:
                st.markdown("**Feature Importance**")
                fi_df = pd.DataFrame({
                    "Fitur": X_combined.columns,
                    "Importance": rf_retrain.feature_importances_,
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(fi_df, use_container_width=True)


# === TAB 4: KOMENTAR PUBLIK ===
with tab4:
    st.subheader("💬 Form Komentar Publik")
    st.write("Tinggalkan komentar, masukan, atau kesan setelah mencoba aplikasi ini.")

    if not mongo_connected:
        st.error(f"❌ Koneksi database bermasalah: {mongo_error}")
    else:
        with st.form("form_komentar_publik"):
            nama_user = st.text_input("Nama Anda")
            isi_komentar = st.text_area("Komentar / Pesan")
            kirim_btn = st.form_submit_button("Kirim Komentar")

            if kirim_btn:
                if nama_user.strip() and isi_komentar.strip():
                    comments_col.insert_one({
                        "nama": nama_user.strip(),
                        "komentar": isi_komentar.strip(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    get_mongo_comments.clear()
                    st.success("✅ Komentar berhasil disimpan ke MongoDB!")
                    st.rerun()
                else:
                    st.warning("⚠️ Mohon isi nama dan komentar terlebih dahulu.")

        st.markdown("---")
        st.markdown("#### 📋 Daftar Komentar Pengunjung")
        daftar_komentar = get_mongo_comments()
        if daftar_komentar:
            st.dataframe(pd.DataFrame(daftar_komentar), use_container_width=True)
        else:
            st.info("Belum ada komentar yang masuk. Jadilah yang pertama berkomentar!")