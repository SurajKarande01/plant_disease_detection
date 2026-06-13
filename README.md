# 🌿 Plant Disease Identification & Diagnosis Platform

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

An end-to-end Machine Learning web application powered by a custom **Convolutional Neural Network (CNN)** and **Streamlit** to instantly classify, identify, and diagnose crop leaf diseases. This project helps farmers, botanists, and agricultural researchers detect plant pathologies early from leaf images.

---

## 📌 Table of Contents

1. [Key Features](#-key-features)
2. [Supported crops & Diseases](#-supported-crops--diseases)
3. [Model Architecture & Training](#-model-architecture--training)
4. [Project Directory Structure](#-project-directory-structure)
5. [Installation & Setup](#-installation--setup)
6. [Usage Guide](#-usage-guide)
7. [Future Enhancements](#-future-enhancements)
8. [License](#-license)

---

## 🚀 Key Features

*   📷 **Instant Inference & Classification:** Upload plant leaf images (`.jpg`, `.jpeg`, `.png`) to identify diseases in milliseconds.
*   🧠 **Deep Learning Classifier:** Custom Sequential CNN architecture optimized for crop classification.
*   🌾 **Multi-Crop Support:** Detects diseases across **Corn (Maize)**, **Grape**, **Soybean**, and **Tomato**.
*   🖥️ **Interactive Web Interface:** Modern, clean, and user-friendly web app built using Streamlit.
*   🖼️ **Real-Time Visual Feedback:** Shows the original image resolution and resizes it live during processing.

---

## 🌾 Supported Crops & Diseases

The model is trained to recognize **15 distinct classes** representing healthy states and specific diseases:

| Crop | Disease / Health Status | Label Class in Dataset |
| :--- | :--- | :--- |
| **🌽 Corn (Maize)** | Gray Leaf Spot (*Cercospora*) | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| **🌽 Corn (Maize)** | Common Rust | `Corn_(maize)___Common_rust_` |
| **🌽 Corn (Maize)** | Northern Leaf Blight | `Corn_(maize)___Northern_Leaf_Blight` |
| **🌽 Corn (Maize)** | **Healthy** | `Corn_(maize)___healthy` |
| **🍇 Grape** | Black Rot | `Grape___Black_rot` |
| **🍇 Grape** | Black Measles (*Esca*) | `Grape___Esca_(Black_Measles)` |
| **🍇 Grape** | Leaf Blight (*Isariopsis*) | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` |
| **🍇 Grape** | **Healthy** | `Grape___healthy` |
| **🌱 Soybean** | Septoria Brown Spot | `Soyabean_Septoria_Brown_Spot` |
| **🌱 Soybean** | Vein Necrosis | `Soyabean_Vein Necrosis` |
| **🌱 Soybean** | **Healthy** | `Soybean___healthy` |
| **🍅 Tomato** | Bacterial Spot | `Tomato___Bacterial_spot` |
| **🍅 Tomato** | Early Blight | `Tomato___Early_blight` |
| **🍅 Tomato** | Late Blight | `Tomato___Late_blight` |
| **🍅 Tomato** | **Healthy** | `Tomato___healthy` |

---

## 🧠 Model Architecture & Training

The core model is a sequential **Convolutional Neural Network (CNN)** built with Keras & TensorFlow.

### CNN Layer Breakdown

| Layer No. | Layer Type | Parameters / Filter Size | Output Shape | Activation | Details |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **-** | **Input** | RGB Leaf Image | `(128, 128, 3)` | - | Normalized to range `[0.0, 1.0]` |
| **1** | **Conv2D** | 32 filters, `3x3` kernel | `(128, 128, 32)` | ReLU | Identifies low-level spatial features |
| **2** | **MaxPooling2D** | Pool size `3x3` | `(42, 42, 32)` | - | Downsamples spatial dimensions |
| **3** | **Conv2D** | 16 filters, `3x3` kernel | `(42, 42, 16)` | ReLU | Extracts higher-level features |
| **4** | **MaxPooling2D** | Pool size `2x2` | `(21, 21, 16)` | - | Downsamples features |
| **5** | **Flatten** | - | `(7056)` | - | Flattens 2D feature maps into a 1D vector |
| **6** | **Dense** | 8 Neurons | `(8)` | ReLU | Fully connected bottleneck layer |
| **7** | **Dense (Output)**| 15 Neurons | `(15)` | Softmax | Outputs probability distributions for 15 classes |

### Training Hyperparameters
*   **Target Image Dimensions:** `128 x 128` pixels (RGB)
*   **Loss Function:** `categorical_crossentropy` (Multi-class setup)
*   **Optimizer:** `Adam` with a customized learning rate of `0.0001`
*   **Training Specs:** `50 Epochs` | `Batch Size = 128`
*   **Dataset Split:** 80% Train (with a 20% validation split on training data) and 20% Test.

---

## 📂 Project Directory Structure

```text
plant_disease_detection/
├── Model/
│   └── plant_disease_model.h5      # Pre-trained CNN model file
├── Test Images/                    # Folder containing leaf sample images for testing
│   └── ...
├── Plant_Disease_Detection.py      # Python script for model training and evaluation
├── main_app.py                    # Streamlit web application code
└── README.md                       # Project Documentation
```

---

## 💻 Installation & Setup

### Prerequisites
Make sure you have **Python 3.8+** installed. We highly recommend using a virtual environment to manage dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/SurajKarande01/plant_disease_detection.git
cd plant_disease_detection
```

### 2. Set Up a Virtual Environment (Recommended)
**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries using pip:
```bash
pip install numpy streamlit opencv-python tensorflow keras matplotlib scikit-learn pandas pillow
```

---

## 🖥️ Usage Guide

### Running the Web Application
To run the Streamlit user interface locally, execute the following command:
```bash
streamlit run main_app.py
```
This will spin up a local server. A browser window will automatically launch at `http://localhost:8501`.

1. **Upload Leaf Image:** Drag & drop or upload any image file from the `Test Images` folder or your own leaf collection.
2. **Predict:** Click the **Predict Disease** button.
3. **View Results:** The model will display the uploaded image and output the identified class label (e.g., *Tomato with Late blight leaf*).

### Training the Model
If you want to re-train the model or modify the architecture:
1. Ensure your plant dataset is structured in a `Dataset` folder containing subdirectories for each of the 15 classes.
2. Update the `dataset_path` variable on line 24 of `Plant_Disease_Detection.py`.
3. Run the script:
   ```bash
   python Plant_Disease_Detection.py
   ```
4. This script will train the CNN, plot accuracy/loss metrics, and save the updated weights as `plant_disease_model.h5`.

---

## 🛠️ Future Enhancements

*   [ ] **Incorporate Transfer Learning:** Integrate pre-trained state-of-the-art models like MobileNetV3 or ResNet50 for higher accuracy.
*   [ ] **Add Treatment Guidelines:** Display organic & chemical treatment options, pesticide guidelines, or preventative measures alongside classification results.
*   [ ] **Weather Integration:** Provide localized agricultural weather alerts and disease risk indexes.
*   [ ] **Mobile Optimization:** Adapt the layout for smoother access via mobile devices in fields.

---

## 📄 License

This project is licensed under the MIT License.
