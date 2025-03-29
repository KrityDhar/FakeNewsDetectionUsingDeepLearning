import numpy as np
import pandas as pd
import tensorflow as tf
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score, roc_curve, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import os


directory = "/content/drive/MyDrive/fakenews"
print("Files in directory:", os.listdir(directory))


authentic_path = f"{directory}/LabeledAuthentic-7K.csv"
fake_path = f"{directory}/LabeledFake-1K.csv"

df_authentic = pd.read_csv(authentic_path)
df_fake = pd.read_csv(fake_path)

df_authentic['label'] = 1  
df_fake['label'] = 0  
df = pd.concat([df_authentic, df_fake], ignore_index=True)
df['text'] = df['headline'].fillna('') + ' ' + df['content'].fillna('')

def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', str(text))  
    text = text.lower()
    return text

df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.strip() != '']


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=100)
y = df['label'].values


print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Resampled dataset shape:", np.bincount(y_resampled))
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


embedding_dim = 300
embedding_path = os.path.join(directory, "cc.bn.300.vec")
embedding_index = {}

if os.path.exists(embedding_path):
    print("Loading pre-trained embeddings...")
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coef

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=100,
                                trainable=False)
else:
    print("Pre-trained embeddings not found! Using random initialized embeddings.")
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                                output_dim=embedding_dim,
                                input_length=100,
                                trainable=True)


model = Sequential([
    embedding_layer,
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(64)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("\nTraining Model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)


y_pred_prob = model.predict(X_test).flatten()  
y_pred = (y_pred_prob >= 0.5).astype(int)  


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Authentic"], yticklabels=["Fake", "Authentic"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


model.save(os.path.join(directory, "banfakenews_bilstm_model.h5"))
print("\nModel saved as banfakenews_bilstm_model.h5")


def predict_fake_news(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]
    return "Authentic News" if prediction > 0.5 else "Fake News"


example_text = "বাংলাদেশের প্রধানমন্ত্রী আজ নতুন প্রকল্প ঘোষণা করেছেন।"
example_prediction = predict_fake_news(example_text)
print("\nExample Prediction:", example_text, "->", example_prediction)
