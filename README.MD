# Tematická analýza
Toto je README pro repozitář k diplomové práci **Text miningové nástroje pro tematickou analýzu česky psaných textů**.

Repozitář obsahuje 2 složky:
- Klasifikace: Obsahuje 3 python soubory pro klasifikaci témat, každý pro jednu knihovnu (NLTK, Scikit-learn, Keras),
- Modelování: Obsahuje 3 python soubory pro modelování témat, každý pro jednu knihovnu (Gensim, Scikit-learn, Tomotopy),

a 2 CSV soubory s výsledky, kterých dosáhl autor v rámci práce.

### Klasifikace
Každý klasifikační soubor obsahuje metody pro provedení předpzracování textů, které jsou napsány speciálně pro datasety
uvedené níže. Při použití jiných zdrojových dat je vhodné metody upravit. Stejně tak je nutné upravit cestu ke zdrojovým
datovým souborům. Pro strojové učení jsou použity klasifikátory: 
- Naivní Bayesovský klasifikátor (NLTK a Sklearn),
- Metoda podpůrných vektorů (NLTK a Sklearn),
- Logistická regrese (NLTK a Sklearn),
- Rozhodovací strom. (NLTK).

#### Knihovny
- NLTK: https://www.nltk.org,
- Scikit-Learn: https://scikit-learn.org,
- Keras: https://keras.io.

### Modelování
V modelovacích souborech je v rámci předpřípravy textu použito pouze odstranění stop slov, neboť byla využita
již předpřipřavená textová data. Při použití jiných zdrojových dat je vhodné optimální metody předpřípravy do kódu
přídat. Stejně tak je nutné upravit cestu ke zdrojovým datovým souborům. Jsou použity implementace modelů:
- Latent Semantic Analysis, někdy Latent Semantic Indexing (Gensim, Sklearn),
- Latent Dirichlet Allocation (Gensim, SKlearn, Tomotopy).

#### Knihovny:
- Gensim: https://radimrehurek.com/gensim,
- Scikit-Learn: https://scikit-learn.org,
- Tomotopy: https://bab2min.github.io/tomotopy.

### Data
- Zdrojová data použita pro klasifikaci jsou dostupná zde: https://corpora.kiv.zcu.cz/sentiment,
- Zdrojová data použita pro modelování jsou dostupná zde: https://github.com/versotym/corpusCzechVerse.