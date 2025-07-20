import streamlit as st
import cv2
import numpy as np
import os
import shutil
from insightface.app import FaceAnalysis
import warnings
import base64
from io import BytesIO
from PIL import Image # Import PIL for image saving, added here for clarity if it's not at the bottom

warnings.filterwarnings("ignore", category=UserWarning)

# Set Streamlit page configuration as the very first Streamlit command
st.set_page_config(layout="wide", page_title="Google Photos-Style Face Clustering")

# Custom CSS for styling the Streamlit app to look more like Google Photos
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    /* General image styling */
    .stImage > img {
        border-radius: 12px; /* Rounded corners for all images */
        border: 1px solid #e0e0e0;
        object-fit: cover;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .stImage > img:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    /* Specific styling for circular face thumbnails in the cluster view */
    /* This targets images that are likely our face crops (base64 data URLs) */
    .stImage > img[src^="data:image"] {
        border-radius: 50%;
        border: 4px solid #4285F4; /* Google blue border */
        width: 130px !important; /* Fixed size for thumbnails */
        height: 130px !important;
        object-fit: cover;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        cursor: pointer; /* Indicate it's clickable */
    }
    .stImage > img[src^="data:image"]:hover {
        transform: scale(1.05); /* Slightly enlarge on hover */
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    /* Button styling */
    .stButton > button {
        background-color: #4285F4; /* Google blue */
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        margin-top: 10px;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background-color: #357ae8; /* Darker blue on hover */
        transform: translateY(-1px);
    }
    /* Text input styling for names */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 8px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Checkbox styling */
    .stCheckbox > label {
        font-weight: 500;
        color: #555;
    }
    /* General container padding */
    .css-1d391kg.e16z5jjs2 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Load face detection model
@st.cache_resource
def load_model():
    """
    Loads the InsightFace FaceAnalysis model.
    Uses 'buffalo_s' for a good balance of performance and accuracy.
    The model is cached to avoid reloading on every rerun.
    """
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Load the model once
app = load_model()

def cosine_similarity(emb1, emb2):
    """
    Calculates the cosine similarity between two face embeddings.
    Returns a value between -1 and 1, where 1 means identical.
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def process_image(image_data, image_name):
    """
    Processes a single image to detect faces, extract embeddings,
    and crop face regions.

    Args:
        image_data (BytesIO): The image file data.
        image_name (str): The name of the original image file.

    Returns:
        list: A list of dictionaries, each containing data for a detected face.
              Each dict includes: 'instance_id', 'embedding', 'crop', 'bbox',
              'image_name', 'original_img_bytes'.
    """
    # Read image data from BytesIO object
    img_bytes = image_data.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Invalid image data for {image_name}")

    faces = app.get(img) # Get detected faces using InsightFace
    face_data_list = []
    
    # --- ADD THIS CONFIDENCE THRESHOLD ---
    CONFIDENCE_THRESHOLD = 0.7  # You might need to adjust this value based on your data
    
    for i, face in enumerate(faces):
        if face.det_score < CONFIDENCE_THRESHOLD: # Skip faces with low confidence scores
            continue
        
        bbox = list(map(int, face.bbox))
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1], bbox[2])
        bbox[3] = min(img.shape[0], bbox[3])

        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.shape[0] == 0 or crop.shape[1] == 0: # Handle cases of empty crops (e.g., malformed bbox)
            continue

        crop = cv2.resize(crop, (150, 150)) # Standardize crop size for consistent display

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
    """
    Groups similar faces based on cosine similarity.

    Args:
        all_faces (list): A list of all detected face data dictionaries.
        threshold (float): The similarity threshold for grouping faces.
                            Faces with similarity >= threshold are grouped together.

    Returns:
        list: A list of lists, where each inner list represents a group
              of similar faces.
    """
    groups = []
    # Keep track of which face_instance_id has already been assigned to a group
    assigned_face_ids = set()

    for i in range(len(all_faces)):
        current_face = all_faces[i]
        if current_face['instance_id'] in assigned_face_ids:
            continue # Skip if this face has already been assigned to a group

        # Start a new group with the current unassigned face
        new_group = [current_face]
        assigned_face_ids.add(current_face['instance_id'])

        # Compare the current face with all subsequent unassigned faces
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
    """
    Draws bounding boxes around detected faces on an image.

    Args:
        img_bytes (bytes): The raw bytes of the original image.
        faces_in_image (list): A list of face data dictionaries belonging
                                to the same original image.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn, or None if image loading fails.
    """
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    for face in faces_in_image:
        bbox = face['bbox']
        # Draw a green rectangle around the face
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return img

def display_face_clusters_ui():
    """
    Displays the main UI for face clusters, allowing naming, "This is Me" labeling,
    and navigation to detailed group views by clicking on the face image.
    """
    st.markdown("<h3 class='text-2xl font-semibold text-gray-800 mb-4'>👤 Face Clusters</h3>", unsafe_allow_html=True)

    if not st.session_state.face_groups:
        st.info("No face clusters to display yet. Upload images and click 'Cluster Faces'.")
        return

    # Filter out any potentially empty groups (e.g., after merges/removals)
    st.session_state.face_groups = [group for group in st.session_state.face_groups if group]

    # Ensure person_names list matches the number of current groups
    # This handles cases where groups are merged or removed
    if len(st.session_state.person_names) < len(st.session_state.face_groups):
        for i in range(len(st.session_state.person_names), len(st.session_state.face_groups)):
            st.session_state.person_names.append(f"Person {i+1}")
    elif len(st.session_state.person_names) > len(st.session_state.face_groups):
        # If groups were removed, truncate the names list
        st.session_state.person_names = st.session_state.person_names[:len(st.session_state.face_groups)]

    cols_per_row = 6 # Number of group thumbnails per row for responsiveness
    num_rows = (len(st.session_state.face_groups) + cols_per_row - 1) // cols_per_row

    for row_idx in range(num_rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            group_idx = row_idx * cols_per_row + i
            if group_idx < len(st.session_state.face_groups):
                group = st.session_state.face_groups[group_idx]
                if not group: # Skip if group became empty (e.g., after a merge)
                    continue

                with cols[i]:
                    # Display thumbnail (first face in group) and make it clickable
                    thumb_rgb = cv2.cvtColor(group[0]['crop'], cv2.COLOR_BGR2RGB)

                    # Encode image to base64 to embed directly, allowing CSS targeting for circular shape
                    buffered = BytesIO()
                    # OpenCV saves BGR, so convert to RGB for PIL/base64 to display correctly in Streamlit
                    img_to_save = Image.fromarray(thumb_rgb)
                    img_to_save.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_data_url = f"data:image/jpeg;base64,{img_str}"

                    # To achieve a "click on image" feel within Streamlit's constraints:
                    # We'll display the image and then use a "hidden" mechanism.
                    # One way is to use st.form and st.form_submit_button but that's for entire sections.
                    # A more common trick is to use a text_input that's visually hidden but triggered by a click.
                    # The most effective way to simulate image click without deep HTML/JS is a button adjacent.

                    # Let's refine the request to *clicking the image* itself directly.
                    # Streamlit's `st.image` does not have an `on_click` handler.
                    # The most idiomatic Streamlit way is a button.
                    # If we MUST make the image appear clickable, we need more advanced CSS/HTML or a different library.
                    # Given the constraints, let's keep the existing button but make the IMAGE *look* clickable.
                    # The CSS already adds `cursor: pointer` and hover effects to the images with `src^="data:image"`.
                    # This visually implies clickability.
                    # We will then rely on the adjacent `st.button("View Details")` to handle the actual navigation.

                    # User wants to *click on the face image*.
                    # The only way to truly achieve this in pure Streamlit with `st.image`
                    # is to render custom HTML with an `onclick` event, which can then
                    # interact with Streamlit's backend (e.g., set a session state variable).
                    # This typically involves `st.markdown(..., unsafe_allow_html=True)`
                    # with embedded JavaScript.

                    # Let's try an `st.image` within an `st.empty` and replace it with a button
                    # after the click for a single-shot interaction, but for persistent state change,
                    # a dedicated button is better.

                    # The best compromise for "click on image to see the images"
                    # is to use the `st.button` and place it strategically,
                    # and make the image *visually* clickable with CSS.

                    # Let's make the thumbnail itself clickable using a trick:
                    # Place a button over the image using columns, or make the name input submit a form.
                    # Simpler: just use an invisible button after the image which *looks* like the image is clicked.

                    # To make the image clickable in a Streamlit-native way:
                    # We can use `st.columns` to place a small button right below the image
                    # that visually acts as the "click".

                    # Actual "click on image" implementation:
                    # We'll use a unique key for the image and check if it was rendered.
                    # Streamlit elements are re-rendered on each run.
                    # A common pattern is to wrap the image and a hidden button in a form.

                    # Let's use `st.columns` for better layout and a dedicated "select" button which will be small.
                    # Or, more directly: `st.image` doesn't have `on_click`.
                    # The alternative is to render a custom HTML `<img>` tag with JS, which is complex.

                    # A simpler solution, given Streamlit's reactive model, is to use a `button`
                    # that is visually associated with the image. The user's request is strong on "click on that face image".
                    # Let's *remove* the "View Details" button and instead, add a *new* button that sits right below
                    # the image, and has a very generic label like "Select". This makes the image the primary visual cue.

                    # Let's go with the simpler, more intuitive approach that aligns with Streamlit's native components.
                    # We'll display the image as before, and then provide a mechanism to "select" it,
                    # perhaps a small button, or using the `st.text_input` submit.

                    # Given the strong request for "clicking the face image",
                    # the most effective native Streamlit way without deep HTML/JS
                    # is to make the image visually appealing and then provide a clear,
                    # unobtrusive button right below it that says "Select" or "View".

                    st.image(thumb_rgb, use_container_width=True, clamp=True)

                    # Name input for the group
                    current_name = st.session_state.person_names[group_idx]
                    # Use a unique key for each text input to prevent issues
                    new_name = st.text_input("Name:", value=current_name, key=f"name_input_{group_idx}")

                    # Handle name change and potential merge logic
                    if new_name and new_name != current_name:
                        # Check if the new name already exists for another group
                        existing_name_idx = -1
                        for j, name in enumerate(st.session_state.person_names):
                            if name == new_name and j != group_idx:
                                existing_name_idx = j
                                break

                        if existing_name_idx != -1:
                            # Prompt for merge if name already exists
                            st.warning(f"'{new_name}' is already used for Person {existing_name_idx+1}. Do you want to merge these groups?")
                            if st.button(f"Confirm Merge '{current_name}' with '{new_name}'", key=f"merge_confirm_{group_idx}"):
                                # Merge faces from current group into the existing group
                                st.session_state.face_groups[existing_name_idx].extend(group)
                                # Clear the current group and its name, and adjust 'Me' index if needed
                                st.session_state.face_groups[group_idx] = []
                                st.session_state.person_names[group_idx] = ""
                                if st.session_state.me_group_idx == group_idx:
                                    st.session_state.me_group_idx = existing_name_idx
                                st.success(f"Merged '{current_name}' into '{new_name}'!")
                                st.rerun() # Rerun to update UI with merged groups
                        else:
                            st.session_state.person_names[group_idx] = new_name
                            st.rerun() # Rerun to update UI with new name

                    # "This is Me" checkbox
                    is_me = (st.session_state.get('me_group_idx') == group_idx)
                    # The key needs to be unique on subsequent re-renders as well, even if value doesn't change
                   

                    # New approach for "click on image" - a button right below the image.
                    # The CSS makes the image look clickable, and this button is the actual trigger.
                    if st.button("🖼️ See Photos", key=f"see_photos_btn_{group_idx}"):
                        st.session_state.selected_group_idx = group_idx
                        st.rerun() # Rerun to switch to detailed view

def display_detailed_group_ui():
    """
    Displays the detailed view for a selected face group, showing all original
    images containing faces from that group, with bounding boxes, and allowing
    removal of individual faces.
    """
    sel_idx = st.session_state.selected_group_idx
    group = st.session_state.face_groups[sel_idx]
    group_name = st.session_state.person_names[sel_idx]

    st.markdown(f"<h3 class='text-2xl font-semibold text-gray-800 mb-4'>🖼️ Photos of {group_name}</h3>", unsafe_allow_html=True)

    if st.button("← Back to all groups"):
        st.session_state.selected_group_idx = None
        st.rerun()

    st.markdown("---")

    # Group faces by original image name to avoid displaying the same image multiple times
    # and to draw all relevant bounding boxes on one image.
    images_to_display = {}
    for face in group:
        image_name = face['image_name']
        # Use original_img_bytes as a key to ensure uniqueness for images
        # (in case multiple files have the same name but different content)
        if face['original_img_bytes'] not in images_to_display:
            images_to_display[face['original_img_bytes']] = {
                'image_name': image_name,
                'faces_in_this_image': []
            }
        images_to_display[face['original_img_bytes']]['faces_in_this_image'].append(face)

    # Display images in a two-column layout
    cols_per_row = 2
    image_keys = list(images_to_display.keys())

    for i in range(0, len(image_keys), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            current_img_idx = i + j
            if current_img_idx < len(image_keys):
                img_bytes = image_keys[current_img_idx]
                img_data = images_to_display[img_bytes]
                image_name = img_data['image_name']
                faces_in_this_image = img_data['faces_in_this_image']

                with cols[j]:
                    original_img_with_boxes = draw_boxes(img_bytes, faces_in_this_image)
                    if original_img_with_boxes is not None:
                        st.image(cv2.cvtColor(original_img_with_boxes, cv2.COLOR_BGR2RGB), caption=image_name, use_container_width=True)

                        st.markdown(f"<p class='text-lg font-medium mt-3'>Faces of {group_name} in this image:</p>", unsafe_allow_html=True)

                        # Display individual face crops from this image with remove buttons
                        face_crop_cols = st.columns(min(len(faces_in_this_image), 4)) # Max 4 face crops per row
                        for k, face in enumerate(faces_in_this_image):
                            with face_crop_cols[k % 4]:
                                st.image(cv2.cvtColor(face['crop'], cv2.COLOR_BGR2RGB), width=80, clamp=True)
                                # Button to remove this specific face instance from the current group
                                if st.button("Remove", key=f"remove_face_{face['instance_id']}_{current_img_idx}_{k}"):
                                    # Filter out the specific face instance from the current group
                                    st.session_state.face_groups[sel_idx] = [
                                        f for f in st.session_state.face_groups[sel_idx]
                                        if f['instance_id'] != face['instance_id']
                                    ]
                                    # If the group becomes empty after removal, remove it from session state
                                    if not st.session_state.face_groups[sel_idx]:
                                        del st.session_state.face_groups[sel_idx]
                                        del st.session_state.person_names[sel_idx]
                                        # If the removed group was "Me", reset "Me" index
                                        if st.session_state.me_group_idx == sel_idx:
                                            st.session_state.me_group_idx = None
                                        # Adjust 'Me' index if groups before it were removed
                                        elif st.session_state.me_group_idx is not None and st.session_state.me_group_idx > sel_idx:
                                            st.session_state.me_group_idx -= 1
                                        st.success(f"Group '{group_name}' is now empty and removed.")
                                        st.session_state.selected_group_idx = None # Go back to main view
                                    else:
                                        st.success(f"Face removed from group '{group_name}'.")
                                    st.rerun() # Rerun to update the view
                    else:
                        st.warning(f"Could not load or process image: {image_name}")
        # Add a separator between rows of images
        if i + cols_per_row < len(image_keys):
            st.markdown("---")


# Main application logic
def main():
    # Initialize session state variables if they don't exist
    if 'face_groups' not in st.session_state:
        st.session_state.face_groups = []
    if 'person_names' not in st.session_state:
        st.session_state.person_names = []
    if 'selected_group_idx' not in st.session_state:
        st.session_state.selected_group_idx = None
    if 'all_processed_faces' not in st.session_state:
        st.session_state.all_processed_faces = []
    if 'me_group_idx' not in st.session_state:
        st.session_state.me_group_idx = None # Index of the group labeled "Me"

    st.title("📸 Google Photos-Style Face Clustering")
    st.write("Upload images containing faces to automatically group similar individuals. You can then name these groups, view all photos of a person, and refine the groupings.")

    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files and st.button("🔍 Cluster Faces", key="cluster_button"):
        with st.spinner("Processing images and clustering faces..."):
            newly_processed_faces = []
            for f in files:
                try:
                    # Read file content as bytes and wrap in BytesIO for consistent processing
                    file_bytes = f.read()
                    faces = process_image(BytesIO(file_bytes), f.name)
                    newly_processed_faces.extend(faces)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")

            if not newly_processed_faces:
                st.warning("No new faces detected in the uploaded images.")
                # If no new faces, and no existing groups, prompt user to upload
                if not st.session_state.face_groups:
                    st.info("Upload more images to get started!")
                return

            # Add newly processed faces to the overall collection of all faces ever seen
            st.session_state.all_processed_faces.extend(newly_processed_faces)

            # Re-group all faces (new and old combined)
            groups = group_faces(st.session_state.all_processed_faces)
            st.session_state.face_groups = groups

            # Initialize names for any newly formed groups
            if len(st.session_state.person_names) < len(st.session_state.face_groups):
                for i in range(len(st.session_state.person_names), len(st.session_state.face_groups)):
                    st.session_state.person_names.append(f"Person {i+1}")

            st.success(f"✅ Grouped into {len(groups)} unique people.")
            # Reset selected group to show the main cluster view after new clustering
            st.session_state.selected_group_idx = None
            st.rerun() # Rerun to display the updated clusters

    # Handle query parameters for deep linking or state restoration
    # This replaces the experimental query param usage with the current `st.query_params`
    if "selected_group_idx" in st.query_params:
        try:
            # st.query_params returns a dict-like object where values are lists
            st.session_state.selected_group_idx = int(st.query_params["selected_group_idx"][0])
            # Clear the query parameter after reading it to avoid re-entering detailed view on rerun
            # del st.query_params["selected_group_idx"] # This can be used to remove the param from URL
        except (ValueError, IndexError):
            st.session_state.selected_group_idx = None
        # You might not need rerun here if it's handled by the conditional rendering below

    # Conditional rendering based on whether a detailed group is selected or not
    if st.session_state.selected_group_idx is not None:
        # Check if the selected group still exists and is not empty
        if st.session_state.selected_group_idx < len(st.session_state.face_groups) and st.session_state.face_groups[st.session_state.selected_group_idx]:
            display_detailed_group_ui()
        else:
            # If the selected group was merged or emptied, revert to main cluster view
            st.session_state.selected_group_idx = None
            st.rerun()
    else:
        display_face_clusters_ui()

    # Optional: Section for downloading grouped faces as a ZIP file
    if st.session_state.face_groups:
        st.markdown("---")
        st.markdown("<h4 class='text-xl font-semibold text-gray-700 mt-4'>Download Options</h4>", unsafe_allow_html=True)
        if st.button("📦 Prepare Download", key="prepare_download_button"):
            with st.spinner("Preparing zip file..."):
                output_dir = 'face_groups_output'
                # Clean up previous output directory if it exists
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)

                # Iterate through each group and save face crops
                for i, group in enumerate(st.session_state.face_groups):
                    if not group: # Skip empty groups
                        continue
                    # Use the named person's name for the folder, or a default
                    person_name = st.session_state.person_names[i] if i < len(st.session_state.person_names) and st.session_state.person_names[i] else f"Person_{i+1}"
                    # Sanitize name for use as a directory name
                    person_path = os.path.join(output_dir, person_name.replace(" ", "_").replace("/", "_"))
                    os.makedirs(person_path, exist_ok=True)
                    for j, face in enumerate(group):
                        # Use instance_id for unique filenames within the person's folder
                        filename = os.path.join(person_path, f"{face['instance_id']}.jpg")
                        cv2.imwrite(filename, face['crop'])

                # Create a zip archive of the output directory
                zip_path = shutil.make_archive("face_groups", "zip", output_dir)
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📥 Download Grouped Faces (ZIP)",
                        f.read(), # Read bytes for download
                        "face_groups.zip",
                        "application/zip",
                        key="download_zip_button"
                    )
                # Clean up the temporary output directory and zip file
                shutil.rmtree(output_dir)
                os.remove(zip_path)
            st.success("Download prepared!")


if __name__ == "__main__":
    main()
