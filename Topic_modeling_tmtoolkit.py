import csv
from multiprocessing import freeze_support
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import json
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from tmtoolkit.topicmod.model_stats import top_n_from_distribution
from tmtoolkit.topicmod.tm_lda import compute_models_parallel


if __name__ == '__main__':
    freeze_support()

    # Příprava dat
    def poemsExtraction2():
        with open("stopwords-cs.txt", "r", encoding="utf-8") as stop_words_file:
            stop_words = stop_words_file.read()

        corpus_folder_path = "datasets/corpusCzechVerse-master/ccv2"
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

    poems = poemsExtraction2()
    print("PŘÍPRAVA DAT: HOTOVO")

    """
    # Seznam unikátních slov
    unique_words_set = set()
    # Iterace přes každý seznam slov ve vašem seznamu seznamů slov
    for poem in documents:
        # Přidání všech slov do množiny unikátních slov
        unique_words_set.update(poem)
    # Převod množiny unikátních slov zpět na seznam
    vocab = np.array(list(unique_words_set))
    print("SLOVNÍK: HOTOVO")
    """


    # Zdroj: https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model
    def get_Cv(model, documents, num_topics):
        topics = model.topic_word_
        top_words_per_topic = top_n_from_distribution(topics, top_n=10, val_labels=vocab)

        top_words = []
        for j in range(0, num_topics, 1):
            top_words_topic = []
            for i in range(0, 10, 1):
                top_words_topic.append(top_words_per_topic[i][j])
            top_words.append(top_words_topic)

        texts = [[word for word in doc] for doc in documents]

        # Vytvoření Gensim Dictionary
        dictionary = corpora.Dictionary(texts)

        # Výpočet skóre koherence
        coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()

        return coherence

    """
    # DTM
    word2id = {word: idx for idx, word in enumerate(set(word for doc in documents for word in doc))} # (slovo, index)
    corpus = [[(word2id[word], 1) for word in doc] for doc in documents] # (slovo, četnost výskytů v dokumentu)
    
    document_term_matrix = np.zeros((len(documents), len(word2id)), dtype=int)
    for doc_idx, doc in enumerate(corpus):
        for word_idx, count in doc:
            document_term_matrix[doc_idx, word_idx] = count
    
    print("DTM MATICE: HOTOVO")
    """


    # Vytvoření BOW matice
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([" ".join(poem) for poem in poems])
    print("BOW MATICE: HOTOVO")


    vocab = np.array(vectorizer.get_feature_names_out())
    print("SLOVNÍK: HOTOVO")

    """
    # Slovník
    vocab = vectorizer.get_feature_names_out()
    print("SLOVNÍK: HOTOVO")
    """
    """
    # Nastavení parametrů pro LDA model
    lda_params = {
        'n_topics': 3,  # Počet témat
        'n_iter': 100  # Počet iterací algoritmu
    }
    
    # Spuštění LDA modelu
    lda_models = tm_lda.compute_models_parallel(
        document_term_matrix.astype(int),
        [lda_params]
    )
        
    # Získání tématických distribucí z modelů LDA
    #lda_topic_distributions = [model[1].topic_word_ for model in lda_models]
    lda_model = lda_models[0][1]
    topic_word_distrib = lda_model.topic_word_
    """
    # Konvertování listu distribucí do jednoho numpy array
    #topic_word_distrib = np.concatenate(lda_model, axis=1)
    """
    lda_model = lda_models[0]
    topic_word_distrib = lda_model[1].topic_word_
    """
    # Převod slovníku do 1D numpy array
    #vocab_array = np.array(list(word2id.keys()))

    """
    # Slovník
    # Inicializace prázdného seznamu pro unikátní slova
    unique_words = []
    # Iterace přes každý seznam slov ve vašem seznamu seznamů slov
    for sublist in documents:
        # Iterace přes každé slovo v aktuálním seznamu slov
        for word in sublist:
            # Přidání slova do seznamu unikátních slov, pokud ještě není v seznamu
            if word not in unique_words:
                unique_words.append(word)

    vocab = np.array(unique_words)
    """




    """
    # Evaluace
    LDA_coherence_score = metric_coherence_gensim(
        topic_word_distrib=topic_word_distrib,
        dtm=document_term_matrix,
        vocab=vocab,
        texts=documents,
        measure='c_v',
        return_mean=True,
        top_n=10
    )
    print("LDA Coherence Score:", LDA_coherence_score)
    """


    # Hledání optimálního nastavení parametrů
    def calc_topics_coherence_scores(min_topics=2, max_topics=100, step=5):
        num_topics_range = range(min_topics, max_topics + 1, step)

        lda_coherence_scores = []
        lda_num_topics_valid_range = []
        max_coherence_score_lda = 0
        lda_coh_score = 0
        max_optimal_num_topics_LDA = 0

        i = 0
        max_coherence_score = 0
        for num_topics in num_topics_range:
            lda_num_topics_valid_range.append(num_topics)

            dtms = {
                "main": X
            }

            lda_params = {
                'n_topics': num_topics,  # Počet témat
                'n_iter': 100  # Počet iterací algoritmu
            }

            lda_models = compute_models_parallel(dtms, [lda_params])

            model = lda_models["main"][0][1]

            LDA_coherence_score = get_Cv(model, poems, num_topics)
            lda_coherence_scores.append(LDA_coherence_score)
            print(num_topics)
            print(LDA_coherence_score)

            # Výpis n nejdůležitějších slov v každém tématu
            #print_ldamodel_topic_words(model.topic_word_, vocab)

            #topic_word_distrib = np.concatenate(lda_model, axis=1)
            """
            LDA_coherence_score = metric_coherence_gensim(
                topic_word_distrib=model.topic_word_,
                dtm=X,
                vocab=vocab,
                texts=documents,
                measure='c_v',
                return_mean=True,
                top_n=10
            )
            
            print(num_topics)
            print(LDA_coherence_score)
            """

            """
            # Ošeření NaN hodnot
            if not np.isnan(LDA_coherence_score):
                lda_coherence_scores.append(LDA_coherence_score)
                lda_num_topics_valid_range.append(num_topics)
            """

            # Ukončení po 2 po sobě jdoucích klesajících hodnotách skóre koherence
            if i >= 2:
                if lda_coherence_scores[i] < lda_coherence_scores[i - 1] and lda_coherence_scores[i - 1] < lda_coherence_scores[i - 2]:
                    optimal_num_topics_LDA = list(num_topics_range)[i - 2]
                    lda_coh_score = lda_coherence_scores[i - 2]
                    break
            i += 1

            # Vrácení max hodnoty, kdyby případ výše nenastal nebyla splněna
            if LDA_coherence_score > max_coherence_score:
                max_coherence_score = LDA_coherence_score
                optimal_num_topics_LDA = num_topics

        # Finální kontrola, zda již předtím nebyla max hodnota koherence
        if lda_coh_score < max_coherence_score_lda:
            optimal_num_topics_LDA = max_optimal_num_topics_LDA


        plt.plot(lda_num_topics_valid_range, lda_coherence_scores, marker='o')
        plt.xlabel('Počet témat')
        plt.ylabel('Skóre koherence')
        plt.xticks(lda_num_topics_valid_range)
        plt.grid(True)
        plt.show()

        return optimal_num_topics_LDA

    n_topics = calc_topics_coherence_scores()
    print("NALEZENÍ OPTIMÁLNÍHO POČTU TÉMAT: HOTOVO")

    def calc_alpha_beta_coherence_scores(min_alpha=0.1, min_beta=0.1, alpha_step=0.1, beta_step=0.1):
        alpha = list(np.arange(min_alpha, 1, alpha_step))
        beta = list(np.arange(min_beta, 1, beta_step))

        max_lda_coherence_score = {"alpha": 0, "beta": 0, "Coherence score:": 0}

        for a in alpha:
            for b in beta:
                dtms = {
                    "main": X
                }

                lda_params = {
                    'n_topics': n_topics,   # počet témat
                    'alpha': a,             # alpha
                    'eta': b,               # beta
                    'n_iter': 100           # počet iterací
                }

                lda_models = compute_models_parallel(dtms, [lda_params])

                model = lda_models["main"][0][1]

                LDA_coherence_score = get_Cv(model, poems, n_topics)

                #lda_topic_distributions = [model[1].topic_word_ for model in lda_models]

                #topic_word_distrib = np.concatenate(lda_topic_distributions, axis=1)
                """
                LDA_coherence_score = metric_coherence_gensim(
                    topic_word_distrib=model.topic_word_,
                    dtm=X,
                    vocab=vocab,
                    texts=documents,
                    measure='c_v',
                    return_mean=True,
                    top_n=10
                )
                """
                print("alpha: ", round(a, 1), " beta: ", round(b, 1), " Coherence score: ", LDA_coherence_score)

                if LDA_coherence_score >= max_lda_coherence_score["Coherence score:"]:
                    max_lda_coherence_score["alpha"] = round(a, 1)
                    max_lda_coherence_score["beta"] = round(b, 1)
                    max_lda_coherence_score["Coherence score:"] = LDA_coherence_score

        print(max_lda_coherence_score)

        return max_lda_coherence_score["alpha"], max_lda_coherence_score["beta"], max_lda_coherence_score["Coherence score:"]

    alpha, beta, lda_coh_score = calc_alpha_beta_coherence_scores()
    print("NALEZENÍ OPTIMÁLNÍCH HODNOT PARAMETRŮ ALPHA A BETA: HOTOVO")
    # {'alpha': 0.6, 'beta': 0.9, 'Coherence score:': 0.520086281134675}


    # Výsledky
    print("LDA MODEL:")
    print("POČET TÉMAT:", n_topics, "ALFA:", alpha, "BETA:", beta)
    print("VÝSLEDNÉ SKÓRE KOHERENCE:", lda_coh_score)


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
            writer.writerow(["Tmtoolkit LDA", lda_coh_score, n_topics, alpha, beta])

        print("VÝSLEDKY ZAPSÁNY")

    except Exception as e:
        print(e)