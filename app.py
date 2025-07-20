import streamlit as st
import cv2
import numpy as np
import os
import shutil
from insightface.app import FaceAnalysis
import warnings
import base64
from io import BytesIO
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(layout="wide", page_title="Google Photos-Style Face Clustering")

@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_model()

def resize_image_if_large(img, max_size=(1600, 1600)):
    h, w = img.shape[:2]
    if h > max_size[1] or w > max_size[0]:
        scale = min(max_size[0] / w, max_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def process_image(image_data, image_name):
    img_bytes = image_data.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Invalid image data for {image_name}")
    img = resize_image_if_large(img)
    faces = app.get(img)
    face_data_list = []
    CONFIDENCE_THRESHOLD = 0.7
    for i, face in enumerate(faces):
        if face.det_score < CONFIDENCE_THRESHOLD:
            continue
        bbox = list(map(int, face.bbox))
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1], bbox[2])
        bbox[3] = min(img.shape[0], bbox[3])
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crop = cv2.resize(crop, (150, 150))
        face_instance_id = f"{image_name}_{i}_{np.random.randint(100000, 999999)}"
        face_data_list.append({
            'instance_id': face_instance_id,
            'embedding': face.normed_embedding,
            'crop': crop,
            'bbox': bbox,
            'image_name': image_name,
            'original_img_bytes': img_bytes
        })
    return face_data_list

def group_faces(all_faces, threshold=0.7):
    groups = []
    assigned_face_ids = set()
    for i in range(len(all_faces)):
        current_face = all_faces[i]
        if current_face['instance_id'] in assigned_face_ids:
            continue
        new_group = [current_face]
        assigned_face_ids.add(current_face['instance_id'])
        for j in range(i + 1, len(all_faces)):
            compare_face = all_faces[j]
            if compare_face['instance_id'] not in assigned_face_ids:
                sim = cosine_similarity(current_face['embedding'], compare_face['embedding'])
                if sim >= threshold:
                    new_group.append(compare_face)
                    assigned_face_ids.add(compare_face['instance_id'])
        groups.append(new_group)
    return groups

def draw_boxes(img_bytes, faces_in_image):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    for face in faces_in_image:
        bbox = face['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return img

def display_face_clusters_ui():
    st.header("👥 Face Clusters")
    if not st.session_state.face_groups:
        st.info("No clusters to show. Upload and cluster faces first.")
        return
    cols_per_row = 6
    for i in range(0, len(st.session_state.face_groups), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(st.session_state.face_groups):
                group = st.session_state.face_groups[idx]
                thumb_rgb = cv2.cvtColor(group[0]['crop'], cv2.COLOR_BGR2RGB)
                cols[j].image(thumb_rgb, use_container_width=True, clamp=True)
                if cols[j].button("🖼️ See Photos", key=f"see_photos_{idx}"):
                    st.session_state.selected_group_idx = idx
                    st.rerun()

def display_detailed_group_ui():
    idx = st.session_state.selected_group_idx
    group = st.session_state.face_groups[idx]
    st.header(f"🖼️ Group {idx+1} Details")
    if st.button("← Back"):
        st.session_state.selected_group_idx = None
        st.rerun()
    images = {}
    for face in group:
        key = face['original_img_bytes']
        if key not in images:
            images[key] = {'image_name': face['image_name'], 'faces': []}
        images[key]['faces'].append(face)
    for img_bytes, data in images.items():
        boxed = draw_boxes(img_bytes, data['faces'])
        if boxed is not None:
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), caption=data['image_name'], use_container_width=True)

def main():
    if 'face_groups' not in st.session_state:
        st.session_state.face_groups = []
    if 'selected_group_idx' not in st.session_state:
        st.session_state.selected_group_idx = None
    if 'all_processed_faces' not in st.session_state:
        st.session_state.all_processed_faces = []
    if 'already_clustered' not in st.session_state:
        st.session_state.already_clustered = False

    st.title("📸 Google Photos-Style Face Clustering")
    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files and not st.session_state.already_clustered:
        newly_processed_faces = []
        progress = st.progress(0, text="Processing...")
        for i, f in enumerate(files):
            try:
                file_bytes = f.read()
                faces = process_image(BytesIO(file_bytes), f.name)
                newly_processed_faces.extend(faces)
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            progress.progress((i + 1) / len(files))
        progress.empty()

        if newly_processed_faces:
            st.session_state.all_processed_faces.extend(newly_processed_faces)
            st.session_state.face_groups = group_faces(st.session_state.all_processed_faces)
            st.success(f"✅ Found {len(st.session_state.face_groups)} face groups!")
        else:
            st.warning("No faces detected.")
        st.session_state.already_clustered = True
        st.rerun()

    if st.session_state.selected_group_idx is not None:
        display_detailed_group_ui()
    else:
        display_face_clusters_ui()

    if st.button("🔄 Reset"):
        st.session_state.face_groups = []
        st.session_state.all_processed_faces = []
        st.session_state.selected_group_idx = None
        st.session_state.already_clustered = False
        st.rerun()

if __name__ == "__main__":
    main()
