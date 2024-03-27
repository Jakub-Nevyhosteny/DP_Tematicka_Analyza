import csv
import json
import os
import numpy as np
import tomotopy as tp


def poemsExtraction():
    with open("../stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
        stop_words = stop_words_file.read()

    corpus_folder_path = "../datasets/corpusCzechVerse-master/ccv"
    corpus_files = os.listdir(corpus_folder_path)

    poems = []
    for corpus_file in corpus_files:
        corpus_file_path = os.path.join(corpus_folder_path, corpus_file)
        with open(corpus_file_path) as file:
            data = json.load(file)

        for poem_data in data:
            poem = []
            poem_body = poem_data['body']
            for line in poem_body:
                for word_info in line:
                    for word in word_info['words']:
                        lemma = word['lemma']
                        if lemma not in stop_words:
                            poem.append(lemma)
            poems.append(poem)
    return poems

poems = poemsExtraction()
print("PŘÍPRAVA DAT: HOTOVO")


# Modelování
num_topics_range = range(2, 100 + 1, 5)
alpha = list(np.arange(0.1, 1, 0.1))
beta = list(np.arange(0.1, 1, 0.1))

coherence_scores = []
max_coherence = {"num topics": 0, "alpha": 0, "beta": 0, "Coherence score": 0}

i = 0
for num_topics in num_topics_range:
    for a in alpha:
        for b in beta:
            # inicializace modelu
            model = tp.LDAModel(k=num_topics, alpha=a, eta=b)

            # Přidání básní do modelu
            for poem in poems:
                model.add_doc(poem)

            # Trénování modelu
            model.train(100)  # Počet iterací trénování

            # Výpočet koherence
            coh = tp.coherence.Coherence(model, coherence='c_v')
            average_coherence = coh.get_score()
            coherence_scores.append(average_coherence)

            """
            # Výpis nejdůležitějších slov pro každé téma
            for i in range(model.k):
                print("Téma {}: {}".format(i, model.get_topic_words(i)))
            """

            print("num topics", num_topics, "alpha:", round(a, 1), "beta:", round(b, 1), "Coherence score",
                  average_coherence)

            if average_coherence >= max_coherence["Coherence score"]:
                max_coherence["num topics"] = num_topics
                max_coherence["alpha"] = round(a, 1)
                max_coherence["beta"] = round(b, 1)
                max_coherence["Coherence score"] = average_coherence

print(max_coherence)
print("MODELOVÁNÍ: HOTOVO")


# Zobrazení výsledků
print("POČET TÉMAT:", max_coherence["num topics"], "ALFA:", max_coherence["alpha"], "BETA:", max_coherence["beta"])
print("VÝSLEDNÉ SKÓRE KOHERENCE:", max_coherence["Coherence score"])
print()


# Export výsledků
column_names = ["Model", "Skóre koherence", "Počet témat", "Alfa", "Beta"]
file_name = "../Modelování_výsledky.csv"

try:
    # Pokud soubor neexistuje, vytvoříme ho a zapíšeme hlavičku
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='', encoding='UTF-8') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

    with open(file_name, mode='a', newline='', encoding='UTF-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Tomotopy LDA", max_coherence["Coherence score"], max_coherence["num topics"],
                         max_coherence["alpha"], max_coherence["beta"]])

    print("VÝSLEDKY ZAPSÁNY")

except Exception as e:
    print(e)