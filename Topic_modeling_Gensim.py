from multiprocessing import freeze_support
import csv
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    freeze_support()


    # PŘÍPRAVA DAT
    def poemsExtraction():
        with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
            stop_words = stop_words_file.read()

        corpus_folder_path = "datasets/corpusCzechVerse-master/ccv"
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


    # Vytvoření Gensim Dictionary
    id2word = Dictionary(poems)
    print("VYTVOŘENÍ SLOVNÍKU: HOTOVO")


    # Převod na Gensim Corpus
    corpus = [id2word.doc2bow(poem) for poem in poems]
    print("BOW REPREZENTACE: HOTOVO")


    # Hledání optimálního nastavení parametrů
    def calc_topics_coherence_scores(min_topics=2, max_topics=100, step=5):
        num_topics_range = range(min_topics, max_topics + 1, step)

        lda_coherence_scores = []
        lsi_coherence_scores = []
        lda_num_topics_valid_range = []
        lsi_num_topics_valid_range = []
        max_coherence_score_lda = 0
        max_coherence_score_lsi = 0
        lda_coh_score = 0
        lsi_coh_score = 0
        max_optimal_num_topics_LDA = 0
        max_optimal_num_topics_LSI = 0

        # LDA
        i = 0
        for num_topics in num_topics_range:
            lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
            lda_coherence_model = CoherenceModel(model=lda_model, texts=poems, dictionary=id2word, coherence='c_v')
            lda_coherence_score = lda_coherence_model.get_coherence()
            lda_coherence_scores.append(lda_coherence_score)
            lda_num_topics_valid_range.append(num_topics)

            print(num_topics)
            print(lda_coherence_score)

            # Ukončení po 2 po sobě jdoucích klesajících hodnotách skóre koherence
            if i >= 2:
                if lda_coherence_scores[i] < lda_coherence_scores[i-1] and lda_coherence_scores[i-1] < lda_coherence_scores[i-2]:
                    optimal_num_topics_LDA = list(num_topics_range)[i-2]
                    lda_coh_score = lda_coherence_scores[i-2]
                    break
            i += 1

            # Vrácení max hodnoty, kdyby případ výše nenastal
            if lda_coherence_score > max_coherence_score_lda:
                max_coherence_score_lda = lda_coherence_score
                max_optimal_num_topics_LDA = num_topics

        # Finální kontrola, zda již předtím nebyla max hodnota koherence
        if lda_coh_score < max_coherence_score_lda:
            optimal_num_topics_LDA = max_optimal_num_topics_LDA


        # LSI
        i = 0
        for num_topics in num_topics_range:
            lsi_model = LsiModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
            lsi_coherence_model = CoherenceModel(model=lsi_model, texts=poems, dictionary=id2word, coherence='c_v')
            lsi_coherence_score = lsi_coherence_model.get_coherence()
            lsi_coherence_scores.append(lsi_coherence_score)
            lsi_num_topics_valid_range.append(num_topics)

            print(num_topics)
            print(lsi_coherence_score)

            # Ukončení po 2 po sobě jdoucích klesajících hodnotách skóre koherence
            if i >= 2:
                if lsi_coherence_scores[i] < lsi_coherence_scores[i-1] and lsi_coherence_scores[i-1] < lsi_coherence_scores[i-2]:
                    optimal_num_topics_LSI = list(num_topics_range)[i-2]
                    lsi_coh_score = lsi_coherence_scores[i-2]
                    break
            i += 1

            # Vrácení max hodnoty, kdyby případ výše nenastal
            if lsi_coherence_score > max_coherence_score_lsi:
                max_coherence_score_lsi = lsi_coherence_score
                max_optimal_num_topics_LSI = num_topics

        # Finální kontrola, zda již předtím nebyla max hodnota koherence
        if lsi_coh_score < max_coherence_score_lsi:
            lsi_coh_score = max_coherence_score_lsi
            optimal_num_topics_LSI = max_optimal_num_topics_LSI


        # Grafy
        plt.plot(lda_num_topics_valid_range, lda_coherence_scores, marker='o')
        plt.title('LDA')
        plt.xlabel('Počet témat')
        plt.ylabel('Skóre koherence')
        plt.xticks(lda_num_topics_valid_range)
        plt.grid(True)
        plt.show()

        plt.plot(lsi_num_topics_valid_range, lsi_coherence_scores, marker='o')
        plt.title('LSI')
        plt.xlabel('Počet témat')
        plt.ylabel('Skóre koherence')
        plt.xticks(lsi_num_topics_valid_range)
        plt.grid(True)
        plt.show()

        return optimal_num_topics_LDA, optimal_num_topics_LSI, lsi_coh_score


    num_topics_LDA, num_topics_LSI, lsi_coh_score = calc_topics_coherence_scores()
    print(num_topics_LDA, num_topics_LSI)
    print("NALEZENÍ OPTIMÁLNÍHO POČTU TÉMAT: HOTOVO")


    def calc_alpha_beta_coherence_score(min_alpha=0.1, min_beta=0.1, alpha_step=0.1, beta_step=0.1):
        alpha = list(np.arange(min_alpha, 1, alpha_step))
        beta = list(np.arange(min_beta, 1, beta_step))
        max_lda_coherence_score = {"alpha": 0, "beta": 0, "Coherence score:": 0}
        for a in alpha:
            for b in beta:
                lda_model = LdaModel(corpus=corpus, num_topics=num_topics_LDA, id2word=id2word, alpha=a, eta=b)

                lda_coherence_model = CoherenceModel(model=lda_model, texts=poems, dictionary=id2word, coherence='c_v')

                lda_coherence_score = lda_coherence_model.get_coherence()

                print("alpha: ", round(a, 1), " beta: ", round(b, 1)," Coherence score: ", lda_coherence_score)

                if lda_coherence_score >= max_lda_coherence_score["Coherence score:"]:
                    max_lda_coherence_score["alpha"] = round(a, 1)
                    max_lda_coherence_score["beta"] = round(b, 1)
                    max_lda_coherence_score["Coherence score:"] = lda_coherence_score

        return max_lda_coherence_score["alpha"], max_lda_coherence_score["beta"], max_lda_coherence_score["Coherence score:"]

    alpha, beta, lda_coh_score = calc_alpha_beta_coherence_score()
    print("NALEZENÍ OPTIMÁLNÍCH HODNOT PARAMETRŮ ALPHA A BETA: HOTOVO")


    # Výsledky
    print("LDA MODEL:")
    print("POČET TÉMAT:", num_topics_LDA, "ALFA:", alpha, "BETA:", beta)
    print("VÝSLEDNÉ SKÓRE KOHERENCE:", lda_coh_score)
    print()
    print("LSI MODEL:")
    print("POČET TÉMAT:", num_topics_LSI)
    print("VÝSLEDNÉ SKÓRE KOHERENCE:", lsi_coh_score)
    print()


    # Export výsledků
    column_names = ["Model", "Skóre koherence", "Počet témat", "Alfa", "Beta"]
    file_name = "Modelování_výsledky.csv"

    try:
        # Pokud soubor neexistuje, vytvoříme ho a zapíšeme hlavičku
        if not os.path.exists(file_name):
            with open(file_name, 'w', newline='', encoding='UTF-8') as file:
                writer = csv.writer(file)
                writer.writerow(column_names)

        with open(file_name, mode='a', newline='', encoding='UTF-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Gensim LDA", lda_coh_score, num_topics_LDA, alpha, beta])
            writer.writerow(["Gensim LSI", lsi_coh_score, num_topics_LSI, "", ""])

        print("VÝSLEDKY ZAPSÁNY")

    except Exception as e:
        print(e)