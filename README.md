# 📦 Image-Based Object Counting App (Streamlit + OpenCV)

## 📌 RecursiveZero Internship Assignment Submission
This project is developed as part of the **RecursiveZero Internship Assignment**.

The goal of this assignment is to build an application that takes an image as input and automatically **detects and counts distinct objects** (such as stacked cloth bundles, thread rolls, fabric stacks, etc.) using **Computer Vision** techniques.

The application is built using **Python, OpenCV, and Streamlit** and provides a clean UI for uploading images and displaying detection results.

---

## 🎯 Problem Statement
- Take an image as input.
- Detect stacked / repeated objects in the image.
- Count total distinct objects.
- Display detected objects with bounding boxes.
- Show output clearly for evaluation.

---

## 🚀 Features Implemented
✅ Upload image through Streamlit UI  
✅ Convert image to binary mask for segmentation  
✅ Noise removal using morphological operations  
✅ Contour detection for object extraction  
✅ Bounding boxes drawn around each detected object  
✅ Final count displayed clearly  
✅ Processed mask shown for verification  
✅ Works for different types of object images  

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV**
- **NumPy**
- **Streamlit**

---

## ⚙️ Approach / Method Used
The object detection & counting is done using the following steps:

1. **Image Upload**
2. Convert to **Grayscale**
3. Apply **Gaussian Blur** (reduce noise)
4. Apply **Thresholding (Binary Mask Generation)**
5. Apply **Morphological Operations** (Opening + Closing)
6. Extract **Contours**
7. Filter small contours (noise removal)
8. Draw **Bounding Boxes**
9. Count valid detected objects
10. Display results on Streamlit UI

---

## 🖼️ Screenshots

### 1. Streamlit UI (Upload Page)

<img width="1920" height="1080" alt="ui-home" src="https://github.com/user-attachments/assets/a822ba21-7985-4dae-839f-b9763f671c42" />
---

### 📊 Sample Output
Example Output shown in the app:

<img width="1920" height="1080" alt="output2" src="https://github.com/user-attachments/assets/d7d9d311-8a1e-416b-ac89-17876ca9fb50" />

Total distinct object count: 10

````

