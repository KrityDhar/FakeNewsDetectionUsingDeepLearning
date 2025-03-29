# FakeNewsDetectionUsingDeepLearning
It is a Bangla Fake News detection system using Customized Bi-LSTM

This repository contains a fake news detection model using a Bidirectional LSTM (BiLSTM) architecture. The model is trained on the BanfakNews dataset, which consists of labeled authentic and fake news articles.

**Features**

Text Preprocessing: Cleans text data by removing special characters and converting text to lowercase.
Tokenization & Padding: Uses Keras' Tokenizer to convert text into sequences and pad them to a fixed length.
Handling Imbalanced Data: Applies SMOTE to balance the dataset.
Pre-trained Word Embeddings: Option to use Bengali FastText embeddings (cc.bn.300.vec).
BiLSTM Model: A deep learning model for classification.
Evaluation Metrics: Reports accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curve.

**Setup Instructions
**
1️⃣ Prerequisites

Ensure you have Python installed (recommended: Python 3.8+). Install the required dependencies using:

pip install numpy pandas tensorflow scikit-learn imbalanced-learn matplotlib seaborn

2️⃣ Clone the Repository

git clone https://github.com/KrityDhar/FakeNewsDetectionUsingDeepLearning.git
cd FakeNewsDetectionUsingDeepLearning

3️⃣ Dataset Placement

Place the dataset files (LabeledAuthentic-7K.csv and LabeledFake-1K.csv) in the data/ directory:

/data
   ├── LabeledAuthentic-7K.csv
   ├── LabeledFake-1K.csv

4️⃣ Run the Script

Execute the training script:

python train.py

This will train the model and save it as banfakenews_bilstm_model.h5.

5️⃣ Making Predictions

After training, you can use the model to classify new text samples. Modify and run:

from model import predict_fake_news

text = "বাংলাদেশের প্রধানমন্ত্রী আজ নতুন প্রকল্প ঘোষণা করেছেন।"
prediction = predict_fake_news(text)
print("Prediction:", prediction)

**Model Performance
**
After training, the model reports the following evaluation metrics:

Accuracy: ~ 91.25%
Precision: ~ 91.50%
Recall: ~ 90.73%
F1 Score: ~ 91.11%
ROC-AUC Score: ~ 97.25%

**Acknowledgements
**
This project is developed as part of Deep Learning course under the supervision of Md. Mynoddin at Rangamati Science and Technology University.
