# Skin Cancer Detection using CNN (Deep Learning + Streamlit)

This project implements a Convolutional Neural Network (CNN) for automated skin cancer classification using the HAM10000 (HMNIST 28×28 RGB) dataset.
The trained model is deployed as an interactive Streamlit web application, allowing users to upload skin lesion images and receive real-time predictions across 7 different cancer classes.

_Link to streamlit app: https://skin-cancerr.streamlit.app/_

---

## Features
AI-powered Skin Lesion Classification

### Classifies images into 7 categories:

- akiec – Actinic keratoses
  
- bcc – Basal cell carcinoma

- bkl – Benign keratosis-like lesions

- df – Dermatofibroma

- nv – Melanocytic nevi

- vasc – Vascular lesions

- mel – Melanoma

### Deep Learning Model (CNN)

- Built with TensorFlow/Keras

- Multiple Conv2D + BatchNorm + MaxPool layers

- Fully connected layers with Dropout

- Trained on 28×28×3 HMNIST images

- Achieved ~70% test accuracy (placeholder)

### Streamlit Web App

- Upload skin lesion images

- Real-time prediction

- Clean UI

- Lightweight and fast inference

- Deployed on Streamlit Cloud

---

### Dataset

The model uses:

- HMNIST 28×28 RGB Dataset (from HAM10000)
Provided by: The International Skin Imaging Collaboration (ISIC)

- Contains 7 classes of skin lesions

- Preprocessed to 28×28 RGB images

- Dataset link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## Model Architecture (Summary)
- Input: (28, 28, 3)
- Conv2D → MaxPooling → BatchNorm
- Conv2D → Conv2D → MaxPooling → BatchNorm
- Conv2D → Conv2D
- Flatten → Dense → BatchNorm → Dropout
- Dense → BatchNorm → Dropout
- Dense → BatchNorm
- Output: 7 neurons (softmax)

Total parameters: ~1 million

---

## Technologies Used

- Python

- TensorFlow / Keras

- NumPy, Pandas

- Matplotlib

- Streamlit (deployment)

---

## Installation
1. Clone the repository
```
git clone https://github.com/your-username/skin-cancer-detection-using-cnn.git
cd skin-cancer-detection-using-cnn
```

3. Install dependencies
```
pip install -r requirements.txt
```

5. Run the Streamlit app
```
streamlit run app.py
```

---

## Usage

- Open the Streamlit interface

- Upload a skin lesion image

- The model preprocesses it to (28×28×3)

- The CNN predicts the most likely cancer type

- Probability scores for all classes are displayed

---

## Results

- Test accuracy: ~70%

- Fast real-time inference

- Performs best on classes with higher dataset representation

- Limitations due to:

  - Low resolution (28×28)
  
  - Class imbalance in HAM10000

---

## Future Improvements

- Train using full-resolution HAM10000 images

- Apply augmentation to reduce class imbalance

- Add Grad-CAM for interpretability

- Use pretrained models (ResNet, EfficientNet)

- Improve UI/UX in Streamlit
