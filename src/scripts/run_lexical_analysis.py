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
    parser.add_argument('--graph', action='store_true', help='Affiche et exporte le graphe global de cooccurrence (matplotlib + Gephi)')
    parser.add_argument('--techlist', type=str, help='Fichier de mots techniques (1/ligne)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Fichier de configuration YAML')
    parser.add_argument('--csv', action='store_true', help='Exporte les statistiques lexicales de tous les documents en CSV + stats globales')
    parser.add_argument('--n-process', type=int, default=1, help='Nombre de processus spaCy (batch lexical stats)')
    args = parser.parse_args()

    # Charger config YAML pour récupérer le dossier de résultats
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    res_dir = config['data'].get('lexical_analysis_dir', 'results/lexical_analysis/')
    os.makedirs(res_dir, exist_ok=True)

    texts = read_texts_from_file(args.input)
    tech_words = read_tech_list(args.techlist) if args.techlist else None

    if args.csv:
        print("\n--- Export CSV et stats globales sur tout le corpus (mode batch spaCy) ---")
        from analysis.lexical_analysis import get_lexical_stats_bulk
        results = get_lexical_stats_bulk(texts, batch_size=50, n_process=args.n_process)
        for i, stats in enumerate(results):
            stats['doc_id'] = i
        # Détection dynamique de toutes les clés (en gardant doc_id en premier)
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        # Ordre logique des colonnes
        ordered_keys = ['doc_id', 'num_sentences', 'num_tokens', 'num_types', 'ttr', 'entropy',
                        'avg_word_length', 'avg_sent_length', 'lexical_density',
                        'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'top_words']
        # Ajoute les clés manquantes à la fin
        for k in sorted(all_keys):
            if k not in ordered_keys:
                ordered_keys.append(k)
        out_path = os.path.join(res_dir, 'lexical_stats_all.csv')
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, '') for k in ordered_keys})
        print(f"CSV exporté : {out_path}")
        import numpy as np
        arr = {k: np.array([r[k] for r in results if k in r and isinstance(r[k], (int, float))]) for k in ordered_keys if k != 'doc_id'}
        print("\n--- Statistiques globales sur le corpus ---")
        for k, v in arr.items():
            if len(v) > 0:
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

    if args.graph:
        print("\n--- Visualisation et export du graphe global de cooccurrence ---")
        G = build_cooc_graph(texts)
        import networkx as nx
        import matplotlib.pyplot as plt
        # Visualisation des 50 nœuds les plus connectés
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
        H = G.subgraph([n for n, _ in top_nodes])
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(H, k=0.2)
        nx.draw(H, pos, with_labels=True, node_size=300, font_size=10, edge_color='gray')
        plt.title("Graphe global de cooccurrence (top 50 noeuds)")
        plt.show()
        # Export GEXF pour Gephi
        gexf_path = os.path.join(res_dir, "cooc_graph_global.gexf")
        nx.write_gexf(G, gexf_path)
        print(f"Graphe global exporté pour Gephi : {gexf_path}")

if __name__ == "__main__":
    main()
