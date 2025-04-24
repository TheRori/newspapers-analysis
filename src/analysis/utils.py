"""
utils.py - utilitaires pour l'analyse de texte (stopwords français, etc)
"""
from functools import lru_cache
from collections.abc import Iterable
import stopwordsiso as stopwordsiso

@lru_cache(maxsize=16)
def get_stopwords(lang=("fr",)):
    """
    Retourne un set de stopwords pour la ou les langues spécifiées (tuple ou str), via stopwordsiso.
    Langues supportées : 'fr', 'de', 'en', etc.
    """
    if isinstance(lang, str):
        langs = (lang,)
    else:
        langs = tuple(lang)
    stop = set()
    for l in langs:
        try:
            stop.update(stopwordsiso.stopwords(l))
        except Exception:
            if l == "fr":
                stop.update(['le','la','les','de','du','des','un','une','et','en','à','au','aux'])
            elif l == "de":
                stop.update(['der','die','das','und','zu','den','von','mit','auf','für','im','ist','des','dem','nicht','ein','eine','als','auch','es','an','am'])
            elif l == "en":
                stop.update(['the','and','to','of','in','that','it','is','was','for','on','as','with','at','by','an','be','this','which','or'])
    return stop

get_french_stopwords = lambda: get_stopwords(("fr",))
get_german_stopwords = lambda: get_stopwords(("de",))
