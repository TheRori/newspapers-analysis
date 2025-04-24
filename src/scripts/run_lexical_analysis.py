"""
Script de lancement pour les analyses lexicales sur un corpus de textes.
Usage :
    python run_lexical_analysis.py --input <fichier_txt> [--cooc] [--techlist <fichier>] [--config <fichier_yaml>] [--csv]

Exemple :
    python run_lexical_analysis.py --input corpus.txt --cooc
"""
import argparse
from pathlib import Path
import os
import yaml
import json
import sys
import csv
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from analysis.lexical_analysis import get_lexical_stats, mark_special_words, build_cooc_graph


def read_texts_from_file(path):
    # Détecte le format JSON ou TXT
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # On prend le champ 'content' si présent, sinon 'original_content' ou ''
        texts = [article.get('content') or article.get('original_content', '') for article in data]
        return [t for t in texts if t and isinstance(t, str)]
    else:
        # Chaque ligne = un document
        with open(path, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def read_tech_list(path):
    with open(path, encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())

def main():
    parser = argparse.ArgumentParser(description="Analyse lexicale d'un corpus")
    parser.add_argument('--input', type=str, required=True, help='Fichier texte, 1 doc/ligne')
    parser.add_argument('--cooc', action='store_true', help='Construit et affiche les cooccurrences')
    parser.add_argument('--techlist', type=str, help='Fichier de mots techniques (1/ligne)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Fichier de configuration YAML')
    parser.add_argument('--csv', action='store_true', help='Exporte les statistiques lexicales de tous les documents en CSV + stats globales')
    args = parser.parse_args()

    # Charger config YAML pour récupérer le dossier de résultats
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    res_dir = config['data'].get('lexical_analysis_dir', 'results/lexical_analysis/')
    os.makedirs(res_dir, exist_ok=True)

    texts = read_texts_from_file(args.input)
    tech_words = read_tech_list(args.techlist) if args.techlist else None

    if args.csv:
        print("\n--- Export CSV et stats globales sur tout le corpus ---")
        results = []
        for i, text in enumerate(texts):
            stats = get_lexical_stats(text)
            stats['doc_id'] = i
            results.append(stats)
            if (i+1) % 500 == 0:
                print(f"{i+1} documents traités...")
        out_path = os.path.join(res_dir, 'lexical_stats_all.csv')
        keys = ['doc_id', 'num_sentences', 'ttr', 'entropy']
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, '') for k in keys})
        print(f"CSV exporté : {out_path}")
        arr = {k: np.array([r[k] for r in results]) for k in keys if k != 'doc_id'}
        print("\n--- Statistiques globales sur le corpus ---")
        for k, v in arr.items():
            print(f"{k}: moyenne={np.mean(v):.3f}, médiane={np.median(v):.3f}, min={np.min(v):.3f}, max={np.max(v):.3f}, std={np.std(v):.3f}")
        return

    print(f"Corpus chargé : {len(texts)} documents")
    print("\n--- Statistiques lexicales (premier document) ---")
    stats = get_lexical_stats(texts[0])
    for k, v in stats.items():
        print(f"{k}: {v}")
    # Sauvegarde stats
    with open(os.path.join(res_dir, 'lexical_stats_doc0.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, allow_unicode=True)

    print("\n--- Marquage spécial (premier document) ---")
    special = mark_special_words(texts[0], technical_words=tech_words)
    special_out = [
        {'token': token, **tags} for token, tags in special if any(tags.values())
    ]
    for entry in special_out:
        print(entry)
    with open(os.path.join(res_dir, 'special_tokens_doc0.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(special_out, f, allow_unicode=True)

    if args.cooc:
        print("\n--- Graphe de cooccurrences (global) ---")
        G = build_cooc_graph(texts)
        print(f"Nombre de noeuds: {G.number_of_nodes()}, arêtes: {G.number_of_edges()}")
        # Affiche les 10 arêtes les plus fortes
        top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
        for w1, w2, data in top_edges:
            print(f"{w1} - {w2} : {data['weight']}")
        # Sauvegarde les arêtes principales
        with open(os.path.join(res_dir, 'top_cooc_edges.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump([
                {'w1': w1, 'w2': w2, 'weight': data['weight']} for w1, w2, data in top_edges
            ], f, allow_unicode=True)

if __name__ == "__main__":
    main()
