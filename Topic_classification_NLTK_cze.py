import csv
import json
import os
import time
import nltk
import pandas as pd
import requests
import simplemma
import stanza
from nltk import apply_features
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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


# Korektor
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
            time.sleep(30)  # Časovač na 30 vteřin

    # Znovuodeslání chybných požadavků
    if len(failed_lines) > 0:
        for line in failed_lines:
            corrected_lines.append(json.loads(line.text).get("result"))
            if len(corrected_lines) % 100 == 0:
                print("PAUSE")
                time.sleep(30)

    df_corrected = pd.DataFrame({'label': [labelName] * len(corrected_lines), 'text': corrected_lines})
    return df_corrected

df_positive = korektorAPI(df_positive['text'], "positive")
df_negative = korektorAPI(df_negative['text'], "negative")
df_neutral = korektorAPI(df_neutral['text'], "neutral")
print("KOREKCE TEXTŮ: DONE")



# regex
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
print("REGEX: HOTOVO")


# Odstranění stopslov
with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
    stop_words = stop_words_file.read()
#preprocessed = preprocessed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df_positive['text'] = df_positive['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df_negative['text'] = df_negative['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df_neutral['text'] = df_neutral['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
print("ODSTRANĚNÍ STOPSLOV: HOTOVO")


# Lemmatizace
def stanza_lemma(dataframe):
    nlp = stanza.Pipeline('cs', processors='tokenize,mwt,pos,lemma')
    df_list = dataframe.tolist()
    lemmatized_texts = []

    for original_sentence in df_list:
        doc = nlp(original_sentence.lower())
        lemmatized_words = []

        for sentence in doc.sentences:
            for word in sentence.words:
                lemmatized_words.append(word.lemma)
            lemmatized_texts.append(' '.join(lemmatized_words))

    lemma_df = pd.DataFrame(lemmatized_texts)
    return lemma_df[0]


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


#preprocessed = stanza_lemma(preprocessed)
#preprocessed = simplemmatizace(preprocessed)
df_positive['text'] = simplemmatizace(df_positive['text'])
df_negative['text'] = simplemmatizace(df_negative['text'])
df_neutral['text'] = simplemmatizace(df_neutral['text'])
print("LEMMATIZACE: HOTOVO")


# Spojení
csfd_reviews = pd.concat([df_positive, df_negative, df_neutral])
#csfd_reviews = pd.concat([df_positive, df_negative])
csfd_reviews = csfd_reviews.sample(frac=1, random_state=42)  # frac=1 shuffles all rows, random_state=42 for reproducibility
print("SPOJENÍ: HOTOVO")

# Extrakce příznaků
enc = LabelEncoder()
label = enc.fit_transform(csfd_reviews['label'])
data = list(zip(csfd_reviews['text'], label))

def extractWordFeatures(preprocessed_texts, num_features):
    all_words = []
    for text in preprocessed_texts:
        words = word_tokenize(text, language='czech')  # Tokenization
        all_words.extend(words)

    # Occurrence counting
    word_freq_dist = nltk.FreqDist(all_words)

    # Select the most common words as features
    word_features = []
    for word, _ in word_freq_dist.most_common(num_features):
        word_features.append(word)

    return word_features

word_features = extractWordFeatures(csfd_reviews['text'], 1500)

def createFeatureSet(data, word_features):
    feature_set = []
    for text, label in data:
        words = word_tokenize(text, language='czech')  # Tokenization
        features = {}
        for word in word_features:
            if word in words:
                features[word] = True
            else:
                features[word] = False
        feature_set.append((features, label))

    return feature_set

feature_set = createFeatureSet(data, word_features)
print("VEKTORIZACE: HOTOVO")


# Rozdělení na trénovací a testovací množinu
training, test = train_test_split(feature_set, test_size=0.25, random_state=1)

# TODO: OTESTOVAT CELÝ TENTO PROCES
# Modely
classifiers = {
#    'K Nearest Neighbors': KNeighborsClassifier(),#
    'Rozhodovací strom': DecisionTreeClassifier(),
#    'Random Forest': RandomForestClassifier(),#
    'Logistická regrese': LogisticRegression(),
#    'SGD Classifier': SGDClassifier(max_iter=100),#
    'Naivní Bayesovský klasifikátor': MultinomialNB(),
    'Metoda podpůrných vektorů': SVC(kernel='linear')
}

# Uložení výsledků
results = {
    'Rozhodovací strom': 0,
    'Logistická regrese': 0,
    'Naivní Bayesovský klasifikátor': 0,
    'Metoda podpůrných vektorů': 0
}

for name, classifier in classifiers.items():
    # Trénování
    nltk_model = SklearnClassifier(classifier)
    nltk_model.train(training)

    accuracy = nltk.classify.accuracy(nltk_model, test)
    print(accuracy)
    results[name] = accuracy

    text_features, labels = zip(*test)
    prediction = nltk_model.classify_many(text_features)

    print(classification_report(labels, prediction))

    # TODO: SCREEN !!!
    print(pd.DataFrame(confusion_matrix(labels, prediction),
                       index=[['actual', 'actual', 'actual'], ['positive', 'negative', 'neutral']],
                       columns=[['predicted', 'predicted', 'predicted'], ['positive', 'negative', 'neutral']]))


# Vypsání výsledků
    print("VÝSLEDNÉ PŘESNOSTI KLASIFIKÁTORŮ:")
    for name, result in results.items():
        print(f"{name}:", result)


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
            for name, result in results.items():
                writer.writerow([f"NLTK {name}", result])

        print("VÝSLEDKY ZAPSÁNY")

    except Exception as e:
        print(e)


"""
# Trénování
nltk_model = SklearnClassifier(MultinomialNB())
nltk_model.train(training)
print("TRÉNOVÁNÍ: HOTOVO")


# Testování a evaluace
accuracy = nltk.classify.accuracy(nltk_model, test)
print(accuracy)

text_features, labels = zip(*test)
prediction = nltk_model.classify_many(text_features)

print(classification_report(labels, prediction))

print(pd.DataFrame(confusion_matrix(labels, prediction),
                   index=[['actual', 'actual', 'actual'], ['positive', 'negative', 'neutral']],
                   columns=[['predicted', 'predicted', 'predicted'], ['positive', 'negative', 'neutral']]))
"""