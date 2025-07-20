import face_recognition
from sklearn.cluster import DBSCAN
import os
from PIL import Image
import streamlit as st

def encode_faces(folder):
    encodings = []
    paths = []
    for img_file in os.listdir(folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_file)
            img = face_recognition.load_image_file(img_path)
            faces = face_recognition.face_locations(img)
            if faces:
                encoding = face_recognition.face_encodings(img, faces)[0]
                encodings.append(encoding)
                paths.append(img_path)
    return encodings, paths

def cluster_faces(encodings):
    clustering = DBSCAN(metric='euclidean', n_jobs=-1, eps=0.5, min_samples=1).fit(encodings)
    return clustering.labels_

def show_clusters(image_paths, labels):
    clusters = {}
    for path, label in zip(image_paths, labels):
        clusters.setdefault(label, []).append(path)

    for label, images in clusters.items():
        st.subheader(f"Cluster {label}")
        cols = st.columns(5)
        for i, img_path in enumerate(images):
            with cols[i % 5]:
                st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)
