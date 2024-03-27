import csv
from multiprocessing import freeze_support
import json
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import CoherenceModel
import gensim.corpora as corpora

if __name__ == '__main__':
    freeze_support()

    # PŘÍPRAVA DAT
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

    # Vytvoření TF-IDF matice pomocí scikit-learn
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(poem) for poem in poems])
    print("TF-IDF MATICE: HOTOVO")


    # Evaluace
    # Upraveno dle: https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model
    def get_Cv(model, documents):
        topics = model.components_

        texts = [[word for word in doc] for doc in documents]

        # Vytvoření Gensim Dictionary
        dictionary = corpora.Dictionary(texts)

        # Získání seznamu unikátních slov
        feature_names = [dictionary[i] for i in range(len(dictionary))]

        # Získání 10 nejdůležitějších slov pro každé téma
        top_words = []
        for topic in topics:
            top_words.append([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])

        # Výpočet skóre koherence
        coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()

        return coherence


    # Hledání optimálního nastavení parametrů
    def calc_topic_coherence_scores(min_topics=2, max_topics=100, step=5):
        num_topics_range = range(min_topics, max_topics + 1, step)

        lda_coherence_scores = []
        lsa_coherence_scores = []
        lda_num_topics_valid_range = []
        lsa_num_topics_valid_range = []
        max_coherence_score_lda = 0
        max_coherence_score_lsa = 0
        lda_coh_score = 0
        lsi_coh_score = 0
        max_optimal_num_topics_LDA = 0
        max_optimal_num_topics_LSA = 0

        # LDA
        i = 0
        for num_topics in num_topics_range:
            lda_model = LatentDirichletAllocation(n_components=num_topics)
            lda_model.fit_transform(X) #TODO: Funguje, když odstraním proměnnou?
            LDA_coherence_score = get_Cv(lda_model, poems)

            print(num_topics)
            print(LDA_coherence_score)

            # Ošeření NaN hodnot
            if not np.isnan(LDA_coherence_score):
                lda_coherence_scores.append(LDA_coherence_score)
                lda_num_topics_valid_range.append(num_topics)

            # Ukončení po 2 po sobě jdoucích menších hodnotách skóre koherence
            if i >= 2:
                if lda_coherence_scores[i] < lda_coherence_scores[i-1] and lda_coherence_scores[i-1] < lda_coherence_scores[i-2]:
                    optimal_num_topics_LDA = list(num_topics_range)[i-2]
                    lda_coh_score = lda_coherence_scores[i - 2]
                    break
            i += 1

            # Vrácení max hodnoty, kdyby případ výše nenastal
            if LDA_coherence_score > max_coherence_score_lda:
                max_coherence_score_lda = LDA_coherence_score
                optimal_num_topics_LDA = num_topics

        # Finální kontrola, zda již předtím nebyla max hodnota koherence
        if lda_coh_score < max_coherence_score_lda:
            optimal_num_topics_LDA = max_optimal_num_topics_LDA


        # LSA
        i = 0
        for num_topics in num_topics_range:
            lsa_model = TruncatedSVD(n_components=num_topics)
            lsa_model.fit_transform(X)
            LSA_coherence_score = get_Cv(lsa_model, poems)

            print(num_topics)
            print(LSA_coherence_score)

            # Ošeření NaN hodnot
            if not np.isnan(LSA_coherence_score):
                lsa_coherence_scores.append(LSA_coherence_score)
                lsa_num_topics_valid_range.append(num_topics)

            # Ukončení po 2 po sobě jdoucích klesajících hodnotách skóre koherence
            if i >= 2:
                if lsa_coherence_scores[i] < lsa_coherence_scores[i - 1] and lsa_coherence_scores[i - 1] < lsa_coherence_scores[i - 2]:
                    optimal_num_topics_LSA = list(num_topics_range)[i - 2]
                    lsi_coh_score = lsa_coherence_scores[i - 2]
                    break
            i += 1

            # Vrácení max hodnoty, kdyby případ výše nenastal
            if LSA_coherence_score > max_coherence_score_lsa:
                max_coherence_score_lsa = LSA_coherence_score
                max_optimal_num_topics_LSA = num_topics

        # Finální kontrola, zda již předtím nebyla max hodnota koherence
        if lsi_coh_score < max_coherence_score_lsa:
            lsi_coh_score = max_coherence_score_lsa
            optimal_num_topics_LSA = max_optimal_num_topics_LSA


        # Grafy
        plt.plot(lda_num_topics_valid_range, lda_coherence_scores, marker='o')
        plt.title('LDA')
        plt.xlabel('Počet témat')
        plt.ylabel('Skóre koherence')
        plt.xticks(lda_num_topics_valid_range)
        plt.grid(True)
        plt.show()

        plt.plot(lsa_num_topics_valid_range, lsa_coherence_scores, marker='o')
        plt.title('LSA')
        plt.xlabel('Počet témat')
        plt.ylabel('Skóre koherence')
        plt.xticks(lsa_num_topics_valid_range)
        plt.grid(True)
        plt.show()

        return optimal_num_topics_LDA, optimal_num_topics_LSA, lsi_coh_score

    lda_num_topics, lsa_num_topics, lsa_coh_score = calc_topic_coherence_scores()
    print(lda_num_topics, lsa_num_topics)
    print("NALEZENÍ OPTIMÁLNÍHO POČTU TÉMAT: HOTOVO")


    def calc_alpha_beta_coherence_scores(min_alpha=0.1, min_beta=0.1, alpha_step=0.1, beta_step=0.1):
        alpha = list(np.arange(min_alpha, 1, alpha_step))
        beta = list(np.arange(min_beta, 1, beta_step))

        max_lda_coherence_score = {"alpha": 0, "beta": 0, "Coherence score:": 0}

        for a in alpha:
            for b in beta:
                lda_model = LatentDirichletAllocation(n_components=lda_num_topics, doc_topic_prior=a, topic_word_prior=b)
                lda_model.fit_transform(X) #TODO: Funguje, když odstraním proměnnou?

                LDA_coherence_score = get_Cv(lda_model, poems)

                print("alpha: ", round(a, 1), " beta: ", round(b, 1), " Coherence score: ", LDA_coherence_score)

                if LDA_coherence_score >= max_lda_coherence_score["Coherence score:"]:
                    max_lda_coherence_score["alpha"] = round(a, 1)
                    max_lda_coherence_score["beta"] = round(b, 1)
                    max_lda_coherence_score["Coherence score:"] = LDA_coherence_score

        return max_lda_coherence_score["alpha"], max_lda_coherence_score["beta"], max_lda_coherence_score["Coherence score:"]

    alpha, beta, lda_coh_score = calc_alpha_beta_coherence_scores()
    print("NALEZENÍ OPTIMÁLNÍCH HODNOT PARAMETRŮ ALPHA A BETA: HOTOVO")


    # Výsledky
    print("LDA MODEL:")
    print("POČET TÉMAT:", lda_num_topics, "ALFA:", alpha, "BETA:", beta)
    print("VÝSLEDNÉ SKÓRE KOHERENCE:", lda_coh_score)
    print()
    print("LSI MODEL:")
    print("POČET TÉMAT:", lsa_num_topics)
    print("VÝSLEDNÉ SKÓRE KOHERENCE:", lsa_coh_score)
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
            writer.writerow(["ScikitLearn LDA", lda_coh_score, lda_num_topics, alpha, beta])
            writer.writerow(["ScikitLearn LSA", lsa_coh_score, lsa_num_topics, "", ""])

        print("VÝSLEDKY ZAPSÁNY")

    except Exception as e:
        print(e)