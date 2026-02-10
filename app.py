import streamlit as st
import cv2
import numpy as np
from object_counter import count_objects


st.set_page_config(page_title="Object Counter", layout="wide")

st.title("📦 Image-Based Object Counting App")
st.write("Upload an image and click **Count Objects** to detect stacked objects.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded Image Preview")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

    if st.button("Count Objects"):
        count, output_img, processed_img = count_objects(image)

        st.success(f"✅ Total distinct object count: {count}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Processed Mask (Binary Image)")
            st.image(processed_img, width=300)

        with col2:
            st.subheader("Detected Objects (Bounding Boxes)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), channels="RGB", width=300)

