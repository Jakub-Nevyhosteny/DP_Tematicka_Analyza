import json
import time

import pandas as pd
import requests
import simplemma
import stanza
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# PŘÍPRAVA DAT
with open("datasets/csfd/positive.txt", "r", encoding="utf-8") as positive_file:
    positive = positive_file.read()
with open("datasets/csfd/negative.txt", "r", encoding="utf-8") as negative_file:
    negative = negative_file.read()

positive_rows = positive.split('\n')
df_positive = pd.DataFrame({'label': 'positive', 'text': positive_rows})

negative_rows = negative.split('\n')
df_negative = pd.DataFrame({'label': 'negative', 'text': negative_rows})


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

print("correcting: DONE")


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

#csfd_reviews['text'] = regexPreprocessing(csfd_reviews['text'])
df_positive['text'] = regexPreprocessing(df_positive['text'])
df_negative['text'] = regexPreprocessing(df_negative['text'])
print("regex preprocessing: DONE")

# Lemma (vytvořil jsem já - možná popsat v DP)

def stanza_lemma(dataframe):
    nlp = stanza.Pipeline('cs', processors='tokenize,mwt,pos,lemma')
    df_list = dataframe.tolist()
    lemmatized_texts = []

    for sentence in df_list:
        doc = nlp(sentence)
        lemmatized_words = []

        for sentence in doc.sentences:
            for word in sentence.words:
                lemmatized_words.append(word.lemma)
            lemmatized_texts.append(' '.join(lemmatized_words))

    lemma_df = pd.DataFrame(lemmatized_texts)
    return lemma_df[0]

#csfd_reviews['text'] = stanza_lemma(csfd_reviews['text'])


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

#csfd_reviews['text'] = simplemmatizace(csfd_reviews['text'])
df_positive['text'] = simplemmatizace(df_positive['text'])
df_negative['text'] = simplemmatizace(df_negative['text'])
print("lemmatization: DONE")


csfd_reviews = pd.concat([df_positive, df_negative])
csfd_reviews = csfd_reviews.sample(frac=1, random_state=42)  # frac=1 shuffles all rows, random_state=42 for reproducibility


# Feature Extraction
enc = LabelEncoder()
label = enc.fit_transform(csfd_reviews['label']) # Transforming "spam" to 1 and "ham" to 0
text = csfd_reviews['text'].to_list() # Because vectorizer wnats to get list as an input

with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
    stop_slova = list(stop_words_file.read())

# BOW
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stop_slova, lowercase=True)
text = vectorizer.fit_transform(text).toarray()

# TF-IDF
tfidfconverter = TfidfTransformer()
text = tfidfconverter.fit_transform(text).toarray()

print("vectorization: DONE")


# TRAINING
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.25, random_state=0)

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)


# TESTING
y_pred = classifier.predict(X_test)


# EVALUATION
print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual'], ['positive', 'negative']],
             columns = [['predicted', 'predicted'], ['positive', 'negative']]))