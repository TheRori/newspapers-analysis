"""
lexical_analysis.py
Module d'analyse lexicale indépendante pour corpus textuels.

Fonctions :
- Statistiques lexicales de base (longueur, TTR, entropie)
- Marquage des sigles, mots étrangers, mots techniques
- Construction de graphe de cooccurrences glissantes

Exemple d'utilisation :
    from analysis.lexical_analysis import get_lexical_stats, mark_special_words, build_cooc_graph
"""
import spacy
from collections import Counter, defaultdict
from scipy.stats import entropy
import re
import networkx as nx
from itertools import combinations
import logging
import time
from .utils import get_stopwords

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Charger spaCy pour le français (assurez-vous d'avoir installé fr_core_news_md)
lazy_nlp = None
def get_nlp():
    global lazy_nlp
    if lazy_nlp is None:
        lazy_nlp = spacy.load("fr_core_news_md")
    return lazy_nlp

def get_lexical_stats(text):
    """
    Calcule des statistiques lexicales de base pour un texte.
    Retourne :
        - num_sentences : nombre de phrases
        - ttr : type-token ratio
        - entropy : entropie de la distribution des mots
    """
    nlp = get_nlp()
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    types = set(tokens)
    ttr = len(types) / len(tokens) if tokens else 0
    num_sentences = sum(1 for _ in doc.sents)
    word_entropy = entropy(list(Counter(tokens).values()), base=2) if tokens else 0
    return {
        'num_sentences': num_sentences,
        'ttr': ttr,
        'entropy': word_entropy
    }

def mark_special_words(text, technical_words=None, technical_pos=None):
    """
    Marque dans le texte :
      - sigles (suite de majuscules)
      - mots étrangers (non français)
      - mots techniques (par liste ou POS)
    Retourne une liste de tuples : (token, {'sigle':bool, 'etranger':bool, 'technique':bool})
    """
    nlp = get_nlp()
    doc = nlp(text)
    results = []
    for token in doc:
        is_sigle = bool(re.match(r"^[A-Z]{2,}$", token.text)) or token.shape_ == "XXX"
        is_etranger = token.lang_ != "fr"
        is_technique = False
        if technical_words and token.text.lower() in technical_words:
            is_technique = True
        if technical_pos and token.pos_ in technical_pos:
            is_technique = True
        results.append((token.text, {
            'sigle': is_sigle,
            'etranger': is_etranger,
            'technique': is_technique
        }))
    return results

def build_cooc_graph(texts, window_size=4, min_freq=2, stopwords=None):
    """
    Construit un graphe de cooccurrences glissantes à partir d'une liste de textes.
    - window_size : taille de la fenêtre glissante
    - min_freq : seuil minimal de fréquence pour créer une arête
    - stopwords : set de stopwords à filtrer (optionnel)
    Retourne un networkx.Graph.
    """
    cooc = defaultdict(int)
    for text in texts:
        tokens = [t for t in text.lower().split() if t.isalpha()]
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        for i in range(len(tokens)):
            for j in range(i+1, min(i+window_size, len(tokens))):
                pair = tuple(sorted((tokens[i], tokens[j])))
                if pair[0] != pair[1]:
                    cooc[pair] += 1
    G = nx.Graph()
    for (w1, w2), freq in cooc.items():
        if freq >= min_freq:
            G.add_edge(w1, w2, weight=freq)
    return G

def get_lexical_stats_bulk(texts, batch_size=50, n_process=1, lang='fr'):
    """
    Calcule les stats lexicales pour une liste de textes en batch (pipeline spaCy désactivée).
    Retourne une liste de dicts (stats par doc).
    """
    start_time = time.time()
    logger.info(f"Démarrage analyse lexicale sur {len(texts)} documents (batch_size={batch_size}, n_process={n_process})")
    
    nlp = get_nlp()
    results = []
    stop_words = get_stopwords(lang)
    with nlp.select_pipes(disable=["ner", "lemmatizer"]):
        logger.info(f"Pipeline spaCy configurée: {', '.join(nlp.pipe_names)}")
        processed = 0
        last_log = 0
        last_time = start_time
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
            # Inclure les stopwords pour tous les calculs sauf top_words
            tokens_all = [token.text.lower() for token in doc if token.is_alpha]
            types = set(tokens_all)
            num_tokens = len(tokens_all)
            num_types = len(types)
            ttr = num_types / num_tokens if num_tokens else 0
            num_sentences = sum(1 for _ in doc.sents)
            word_entropy = entropy(list(Counter(tokens_all).values()), base=2) if tokens_all else 0
            avg_word_length = sum(len(t) for t in tokens_all) / num_tokens if num_tokens else 0
            avg_sent_length = num_tokens / num_sentences if num_sentences else 0
            pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
            lexical_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
            lexical_count = sum(pos_counts.get(pos, 0) for pos in lexical_pos)
            lexical_density = lexical_count / num_tokens if num_tokens else 0
            # Exclure les stopwords uniquement pour top_words
            tokens_no_stop = [t for t in tokens_all if t not in stop_words]
            top_words = Counter(tokens_no_stop).most_common(5)
            stats = {
                'num_sentences': num_sentences,
                'num_tokens': num_tokens,
                'num_types': num_types,
                'ttr': ttr,
                'entropy': word_entropy,
                'avg_word_length': avg_word_length,
                'avg_sent_length': avg_sent_length,
                'lexical_density': lexical_density,
                'pos_NOUN': pos_counts.get('NOUN', 0) / num_tokens if num_tokens else 0,
                'pos_VERB': pos_counts.get('VERB', 0) / num_tokens if num_tokens else 0,
                'pos_ADJ': pos_counts.get('ADJ', 0) / num_tokens if num_tokens else 0,
                'pos_ADV': pos_counts.get('ADV', 0) / num_tokens if num_tokens else 0,
                'top_words': ','.join([f"{w}:{c}" for w, c in top_words])
            }
            results.append(stats)
            processed += 1
            if processed % 500 == 0 or processed == len(texts):
                now = time.time()
                batch_docs = processed - last_log
                batch_time = now - last_time if now > last_time else 1e-6
                batch_speed = batch_docs / batch_time
                global_speed = processed / (now - start_time)
                logger.info(f"Traités {processed}/{len(texts)} documents ({processed/len(texts)*100:.1f}%) - {batch_speed:.1f} docs/sec (batch), {global_speed:.1f} docs/sec (global)")
                last_log = processed
                last_time = now
    elapsed = time.time() - start_time
    logger.info(f"Analyse lexicale terminée en {elapsed:.1f} secondes ({len(texts)/elapsed:.1f} docs/sec)")
    return results

def mark_special_words_fast(text, technical_words=None, technical_pos=None):
    """
    Version optimisée du marquage spécial : préfiltre, conditions regroupées.
    """
    nlp = get_nlp()
    with nlp.select_pipes(disable=["ner", "parser", "lemmatizer"]):
        doc = nlp(text)
        results = []
        for token in doc:
            if not token.is_alpha or len(token.text) <= 1:
                continue
            txt = token.text
            is_sigle = (txt.isupper() and len(txt) > 1) or token.shape_ == "XXX"
            is_etranger = token.lang_ != "fr"
            is_technique = (
                (technical_words and txt.lower() in technical_words)
                or (technical_pos and token.pos_ in technical_pos)
            )
            if is_sigle or is_etranger or is_technique:
                results.append((txt, {
                    'sigle': is_sigle,
                    'etranger': is_etranger,
                    'technique': is_technique
                }))
    return results

def build_cooc_matrix_vectorizer(texts, min_df=5, lang='fr'):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    stop_words = get_stopwords(lang)
    vectorizer = CountVectorizer(min_df=min_df, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    cooc_matrix = (X.T @ X).toarray()
    np.fill_diagonal(cooc_matrix, 0)  # ignore auto-cooc
    return cooc_matrix, vocab
