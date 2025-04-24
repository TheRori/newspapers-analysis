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

def build_cooc_graph(texts, window_size=4, min_freq=2):
    """
    Construit un graphe de cooccurrences glissantes à partir d'une liste de textes.
    - window_size : taille de la fenêtre glissante
    - min_freq : seuil minimal de fréquence pour créer une arête
    Retourne un networkx.Graph.
    """
    cooc = defaultdict(int)
    for text in texts:
        tokens = [t for t in text.lower().split() if t.isalpha()]
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
