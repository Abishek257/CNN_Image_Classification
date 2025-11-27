# CNN_Image_Classification
A complete deep-learning project that classifies CIFAR-10 images using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) built with TensorFlow & Keras. This project demonstrates data preprocessing, model building, training, evaluation, and prediction — ideal for beginners and learners in Machine Learning & Deep Learning.

**🚀 Features**

✔️ CIFAR-10 dataset loading & preprocessing

✔️ Data visualization (sample images)

✔️ ANN model for baseline accuracy

✔️ CNN model for high-accuracy classification

✔️ Confusion matrix & classification report

✔️ Visualization of model predictions

**🧠 Models Used**

**1. ANN (Artificial Neural Network)**

Flatten → Dense(300) → Dense(100) → Output(10)

Useful as a baseline model.

**2. CNN (Convolutional Neural Network)**

Conv2D → MaxPooling → Conv2D → MaxPooling → Dense → Output

Provides significantly better accuracy on image data.

**📁 Project Structure**

├── Image_Classification.ipynb

└── README.md

**🛠️ Technologies Used**

Python 

TensorFlow / Keras

NumPy

Matplotlib & Seaborn

Scikit-learn

**📊 Results**

| Model | Accuracy | Notes |
|-------|----------|-------|
| **ANN** | Baseline | Not ideal for images but useful for comparison |
| **CNN** | Higher accuracy | Learns spatial features effectively |

**🧪 How to Run**

**Clone this repository:**

git clone https://github.com/Abishek257/CNN_Image_Classification.git


**Open the notebook:**

jupyter notebook Image_Classification.ipynb


Run all cells.

**📦 Dataset**

This project uses the CIFAR-10 dataset included directly within TensorFlow:

```python
from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
