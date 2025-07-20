import streamlit as st
from utils import encode_faces, cluster_faces, show_clusters
import tempfile
import zipfile
import os

st.title("👥 Face Image Clustering")

uploaded_file = st.file_uploader("Upload a ZIP of images", type=["zip"])
if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        st.info("Encoding faces...")
        encodings, paths = encode_faces(tmpdir)

        if not encodings:
            st.warning("No faces found in uploaded images.")
        else:
            st.success(f"Found {len(encodings)} face(s). Clustering...")

            labels = cluster_faces(encodings)

            st.write(f"🧠 Clustered into {len(set(labels))} groups")
            show_clusters(paths, labels)
