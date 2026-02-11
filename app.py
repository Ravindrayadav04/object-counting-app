import streamlit as st
import cv2
import numpy as np
from PIL import Image
from object_counter import count_cloth_stacks

st.set_page_config(page_title="Cloth Stack Counter", layout="wide")

st.title("🧵 Cloth Stack Object Counter (ROI + Watershed)")
st.write("Detects only the main front stack and counts each cloth piece.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(pil_img, caption="Original Image", width=450)

    if st.button("Count Objects"):
        count, processed_img, output_img = count_cloth_stacks(img_bgr)

        st.success(f"✅ Detected Count: {count}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Processed Mask")
            st.image(processed_img, width=300)

        with col2:
            st.subheader("Detected Objects (Bounding Boxes)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), channels="RGB", width=300)
