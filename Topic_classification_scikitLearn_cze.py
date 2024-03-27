import csv
import os
import pandas as pd
import simplemma
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# NAČTENÍ DAT
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
print("NAČTENÍ DAT: HOTOVO")


# Regex
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


# Lemmatizace
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
print("LEMMATIZACE: HOTOVO")


csfd_reviews = pd.concat([df_positive, df_negative, df_neutral])
csfd_reviews = csfd_reviews.sample(frac=1, random_state=42)  # frac=1 zamíchá všechny řádky
print("SPOJENÍ: HOTOVO")


# Extrakce příznaků
enc = LabelEncoder()
label = enc.fit_transform(csfd_reviews['label'])
text = csfd_reviews['text'].to_list() # Protože vectorizer chce list jako input

with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
    stop_slova = list(stop_words_file.read())

# BOW & Odstranění stop slov & lowercase
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stop_slova, lowercase=True)
text = vectorizer.fit_transform(text).toarray()

# TF-IDF
tfidfconverter = TfidfTransformer()
text = tfidfconverter.fit_transform(text).toarray()
print("VEKTORIZACE: HOTOVO")


# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.25, random_state=0)


# Modely
classifiers = {
    #'Rozhodovací strom': DecisionTreeClassifier(), # V závislosti na výkonnosti stroje TODO: Napsat do DP do výsledků
    'Logistická regrese': LogisticRegression(max_iter=1000), # Možná bude nutné upravit parametr podle velikosti dat
    'Naivní Bayesovský klasifikátor': MultinomialNB(),
    'Metoda podpůrných vektorů': SVC()
}

# Uložení výsledků
results = {
    #'Rozhodovací strom': 0,
    'Logistická regrese': 0,
    'Naivní Bayesovský klasifikátor': 0,
    'Metoda podpůrných vektorů': 0
}

for name, classifier in classifiers.items():
    # Inicializace
    classifier = classifier

    # Trénování
    classifier.fit(X_train, y_train)

    # Testování
    y_pred = classifier.predict(X_test)

    # Evaluace
    results[name] = accuracy_score(y_test, y_pred)
    print(name)
    print(accuracy_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
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
            writer.writerow([f"ScikitLearn {name}", result])

    print("VÝSLEDKY ZAPSÁNY")

except Exception as e:
    print(e)

"""
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)


# TESTING
y_pred = classifier.predict(X_test)


# EVALUATION
print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual'], ['positive', 'negative']],
             columns = [['predicted', 'predicted'], ['positive', 'negative']]))
"""