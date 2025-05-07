"""
Utilitaires pour le filtrage des journaux dans l'application.
Ce module fournit des fonctions pour regrouper les journaux avec des numéros
et pour filtrer les articles en fonction des journaux sélectionnés.
"""

import re

def group_newspapers(articles):
    """
    Regroupe les journaux qui ont des numéros à la fin (ex: L'impartial 2, L'impartial 3)
    sous un nom de base commun.
    
    Args:
        articles: Liste des articles à analyser
        
    Returns:
        Liste triée des noms de journaux regroupés
    """
    newspapers = set()
    newspaper_base_names = {}
    
    # Première passe : collecter tous les noms de journaux
    for article in articles:
        journal_name = None
        if 'newspaper' in article and article['newspaper']:
            journal_name = article['newspaper']
        elif 'journal' in article and article['journal']:
            journal_name = article['journal']
        
        if journal_name:
            newspapers.add(journal_name)
            
            # Extraire le nom de base (sans numéro à la fin)
            base_name_match = re.match(r'(.+?)\s+\d+\.?$', journal_name)
            if base_name_match:
                base_name = base_name_match.group(1)
                if base_name not in newspaper_base_names:
                    newspaper_base_names[base_name] = []
                newspaper_base_names[base_name].append(journal_name)
    
    # Décider quels journaux regrouper
    final_newspapers = set()
    for journal in newspapers:
        # Vérifier si ce journal fait partie d'un groupe
        is_grouped = False
        for base_name, variants in newspaper_base_names.items():
            if journal in variants and len(variants) > 1:
                # Ce journal fait partie d'un groupe, on utilise le nom de base
                final_newspapers.add(base_name)
                is_grouped = True
                break
        
        # Si ce n'est pas un journal groupé, l'ajouter tel quel
        if not is_grouped:
            final_newspapers.add(journal)
    
    return sorted(list(final_newspapers))

def filter_articles_by_journals(articles, selected_journals):
    """
    Filtre les articles en fonction des journaux sélectionnés,
    en tenant compte des journaux groupés.
    
    Args:
        articles: Liste des articles à filtrer
        selected_journals: Liste des journaux sélectionnés
        
    Returns:
        Liste des articles filtrés
    """
    if not selected_journals:
        return articles
    
    filtered_articles = []
    
    for article in articles:
        journal_name = article.get('journal') or article.get('newspaper')
        if not journal_name:
            continue
        
        # Vérifier si le journal correspond à un des journaux sélectionnés
        # ou à un journal avec numéro qui correspond à un nom de base sélectionné
        match_found = False
        
        # Vérification directe
        if journal_name in selected_journals:
            match_found = True
        else:
            # Vérification pour les journaux avec numéro
            base_name_match = re.match(r'(.+?)\s+\d+\.?$', journal_name)
            if base_name_match:
                base_name = base_name_match.group(1)
                if base_name in selected_journals:
                    match_found = True
        
        if match_found:
            filtered_articles.append(article)
    
    return filtered_articles
