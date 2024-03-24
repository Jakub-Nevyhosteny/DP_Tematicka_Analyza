import csv
import json
import os
import time

import simplemma
import stanza
import keras
import numpy as np
import pandas as pd
import requests
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from keras import metrics
from sklearn.metrics import ConfusionMatrixDisplay


# PŘÍPRAVA DAT
with open("datasets/csfd/positive.txt", "r", encoding="utf-8") as positive_file:
    positive = positive_file.read()
with open("datasets/csfd/negative.txt", "r", encoding="utf-8") as negative_file:
    negative = negative_file.read()
with open("datasets/csfd/neutral.txt", "r", encoding="utf-8") as neutral_file:
    neutral = neutral_file.read()

positive_rows = positive.split('\n')
df_positive = pd.DataFrame({'label': 'positive', 'text': positive_rows})

negative_rows = negative.split('\n')
df_negative = pd.DataFrame({'label': 'negative', 'text': negative_rows})

neutral_rows = neutral.split('\n')
df_neutral = pd.DataFrame({'label': 'neutral', 'text': neutral_rows})


def korektorAPI(text, labelName):
    url = "http://lindat.mff.cuni.cz/services/korektor/api/correct?data="
    corrected_lines = []
    failed_lines = []

    for row in text:
        response = requests.get(url + row)
        if response.status_code == 200:
            corrected_lines.append(json.loads(response.text).get("result"))
        else:
            print("Nepodařilo se načíst webovou stránku. Stavový kód:", response.status_code)
            failed_lines.append(json.loads(response.text).get("result"))
            print(row)

        # Spuštění časovače po každých 100 odeslaných požadavcích
        if len(corrected_lines) % 100 == 0:
            print("PAUSE")
            time.sleep(30) # Časovač na 30 vteřin

    # Znovuodeslání chybných požadavků
    if len(failed_lines) > 0:
        for line in failed_lines:
            corrected_lines.append(json.loads(line.text).get("result"))
            if len(corrected_lines) % 100 == 0:
                print("PAUSE")
                time.sleep(30)

    df_corrected = pd.DataFrame({'label': [labelName] * len(corrected_lines), 'text': corrected_lines})
    print(df_corrected)
    return df_corrected




def regexPreprocessing(df_column):
    # Odstranění emailových dadres
    df_column = df_column.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', '', regex=True)
    # Odstranění URLs
    df_column = df_column.str.replace(r'^https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', '', regex=True)
    # Odstranění telefonních čísel
    df_column = df_column.str.replace(r'(\+420\s?\d{3}\s?\d{3}\s?\d{3}|\b\d{3}\s?\d{3}\s?\d{3}\b)', '', regex=True)
    # Odstranění více mezer mezi slov a nahrazení jednou mezerou
    df_column = df_column.str.replace(r'\s+', ' ', regex=True)
    # Odstranění interpunkčních znamének
    df_column = df_column.str.replace(r'[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]', "", regex=True)

    return df_column

df_positive['text'] = regexPreprocessing(df_positive['text'])
df_negative['text'] = regexPreprocessing(df_negative['text'])
df_neutral['text'] = regexPreprocessing(df_neutral['text'])
print("Regex preprocessing: DONE")


# Remove stopwords
with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
    stop_words = stop_words_file.read()
df_positive['text'] = df_positive['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df_negative['text'] = df_negative['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df_neutral['text'] = df_neutral['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
print("STOPWORDS removing: DONE")


# Lemma
def stanza_lemma(dataframe):
    nlp = stanza.Pipeline('cs', processors='tokenize,mwt,pos,lemma')
    df_list = dataframe.tolist()
    lemmatized_texts = []

    for sentence in df_list:
        doc = nlp(sentence.lower())
        lemmatized_words = []

        for sentence in doc.sentences:
            for word in sentence.words:
                lemmatized_words.append(word.lemma)
            lemmatized_texts.append(' '.join(lemmatized_words))

    lemma_df = pd.DataFrame(lemmatized_texts)
    return lemma_df[0]

# Z NĚJAKÉHO DŮVODU, JSOU LEPŠÍ VÝSLEDKY BEZ LEMATIZACE
def simplemmatizace(df):
    df_list = df.tolist()
    lemmaSent = []
    for sentence in df_list:
        lemmaWords = []
        tokens = sentence.split()
        for token in tokens:
            lemmaWords.append(simplemma.lemmatize(token, lang='cs'))
        lemmaSent.append(' '.join(lemmaWords))
    lemma_df = pd.DataFrame(lemmaSent)
    return lemma_df[0]



df_positive['text'] = simplemmatizace(df_positive['text'])
df_negative['text'] = simplemmatizace(df_negative['text'])
df_neutral['text'] = simplemmatizace(df_neutral['text'])
print("Lemmatization: DONE")


# SPOJENÍ
csfd_reviews = pd.concat([df_positive, df_negative, df_neutral])
csfd_reviews = csfd_reviews.sample(frac=1, random_state=42)  # frac=1 shuffles all rows, random_state=42 for reproducibility


enc = LabelEncoder()
labels = enc.fit_transform(csfd_reviews['label']) # Transforming "positive" to 1 and "negative" to 0
one_hot_labels = to_categorical(labels, num_classes=3)
reviews_text = list(csfd_reviews['text'])

"""
# Tokenization and vectorization
tokenizer = Tokenizer(num_words=1500)
tokenizer.fit_on_texts(reviews_text)
texts = tokenizer.texts_to_sequences(reviews_text)

maxlen = max(len(text) for text in texts)
texts = tf.keras.utils.pad_sequences(texts, maxlen=maxlen)

"""
def tav(texts):
    # Tokenizace a vektorizace textu
    tokenizer = Tokenizer(num_words=1500)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding sekvencí
    maxlen = []
    for text in sequences:
        maxlen.append(len(text))
    maxlen = max(maxlen)
    padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=maxlen)

    return padded_sequences, maxlen

texts, maxlen = tav(reviews_text)
print("vectorization: DONE")

# Model
model = Sequential()
model.add(Embedding(1500, 64, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(texts, one_hot_labels, test_size=0.25, random_state=0)


# Training and testing the model
history = model.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_test,y_test))
#model.fit(X_train, y_train, epochs=2, batch_size=512) #2


# Vytvoření grafu přesnosti v průběhu trénování
plt.plot(range(1, 10 + 1), history.history['acc'], label='Trénovací přesnost')
plt.plot(range(1, 10 + 1), history.history['val_acc'], label='Testovací přesnost')
plt.xlabel('Počet opakování')
plt.ylabel('Přesnost')
plt.legend()
plt.show()


y_pred = model.predict(X_test, verbose=False, batch_size=512)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=False, batch_size=512)
print(accuracy)
# Convert one-hot encoded labels back to single class labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print(classification_report(y_test_labels, y_pred_labels))

print(pd.DataFrame(confusion_matrix(y_test_labels, y_pred_labels),
                   index=[['actual', 'actual', 'actual'], ['positive', 'negative', 'neutral']],
                   columns=[['predicted', 'predicted', 'predicted'], ['positive', 'negative', 'neutral']]))
print(accuracy_score(y_test_labels, y_pred_labels))
# TODO: KDE JE ACCURACY

# TODO: VYPSÁNÍ VÝSLEDKŮ

# Export výsledků
column_names = ["Klasifikátor", "Přesnost"]
file_name = "Klasifikace_výsledky.csv"

try:
    # Pokud soubor neexistuje, vytvoříme ho a zapíšeme hlavičku
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='', encoding='UTF-8') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

    with open(file_name, mode='a', newline='', encoding='UTF-8') as file:
        writer = csv.writer(file)
        writer.writerow([f"Keras Neuronové sítě", ]) # TODO

    print("VÝSLEDKY ZAPSÁNY")

except Exception as e:
    print(e)