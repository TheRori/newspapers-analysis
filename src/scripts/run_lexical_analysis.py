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
from analysis.utils import get_stopwords
import networkx as nx

def read_texts_from_file(path):
    """
    Lit les textes depuis un fichier JSON ou TXT.
    Pour JSON, retourne une liste de tuples (id, texte).
    Pour TXT, retourne une liste de tuples (index, texte).
    """
    # Détecte le format JSON ou TXT
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # On prend le champ 'content' si présent, sinon 'original_content' ou ''
        result = []
        for i, article in enumerate(data):
            content = article.get('content') or article.get('original_content', '')
            # Utiliser id ou base_id comme identifiant, sinon un index numérique
            doc_id = article.get('id') or article.get('base_id') or f"doc_{i}"
            if content and isinstance(content, str):
                result.append((doc_id, content))
        return result
    else:
        # Chaque ligne = un document
        with open(path, encoding='utf-8') as f:
            return [(f"doc_{i}", line.strip()) for i, line in enumerate(f) if line.strip()]

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
    parser.add_argument('--stopwords-cooc', action='store_true', help='Filtrer les stopwords dans le graphe de cooccurrence')
    parser.add_argument('--min-edge-weight', type=int, default=0, help="Poids minimal des arêtes à conserver dans le graphe de cooccurrence")
    args = parser.parse_args()

    # Charger config YAML pour récupérer le dossier de résultats
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    res_dir = config['data'].get('lexical_analysis_dir', 'results/lexical_analysis/')
    # Chemin robuste relatif à la racine projet si non absolu
    from pathlib import Path
    def get_project_root():
        # Cherche la racine du projet (là où est config.yaml)
        return Path(__file__).resolve().parents[2]
    project_root = get_project_root()
    if not os.path.isabs(res_dir):
        res_dir = project_root / res_dir
    os.makedirs(res_dir, exist_ok=True)

    def get_save_path(filename):
        if os.path.isabs(filename):
            return filename
        return str(res_dir / filename)
    texts = read_texts_from_file(args.input)
    tech_words = read_tech_list(args.techlist) if args.techlist else None
    stopwords = set()
    if args.stopwords_cooc:
        stopwords = get_stopwords(('fr',))

    # --- Traitement du corpus complet pour stats globales et logs (toujours exécuté) ---
    print(f"Corpus chargé : {len(texts)} documents")
    print("\n--- Statistiques lexicales (premier document) ---")
    stats = get_lexical_stats(texts[0][1])
    for k, v in stats.items():
        print(f"{k}: {v}")
    # Sauvegarde stats
    with open(get_save_path('lexical_stats_doc0.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, allow_unicode=True)

    print("\n--- Marquage spécial (premier document) ---")
    special = mark_special_words(texts[0][1], technical_words=tech_words)
    special_out = [
        {'token': token, **tags} for token, tags in special if any(tags.values())
    ]
    for entry in special_out:
        print(entry)
    with open(get_save_path('special_tokens_doc0.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(special_out, f, allow_unicode=True)

    # --- Si CSV ou pas, faire stats globales sur tout le corpus ---
    from analysis.lexical_analysis import get_lexical_stats_bulk
    results = get_lexical_stats_bulk([t[1] for t in texts], batch_size=50, n_process=args.n_process)
    for i, s in enumerate(results):
        s['doc_id'] = texts[i][0]
    # Détection dynamique de toutes les clés (en gardant doc_id en premier)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    ordered_keys = ['doc_id', 'num_sentences', 'num_tokens', 'num_types', 'ttr', 'entropy',
                    'avg_word_length', 'avg_sent_length', 'lexical_density',
                    'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'top_words']
    for k in sorted(all_keys):
        if k not in ordered_keys:
            ordered_keys.append(k)
    import numpy as np
    arr = {k: np.array([r[k] for r in results if k in r and isinstance(r[k], (int, float))]) for k in ordered_keys if k != 'doc_id'}
    print("\n--- Statistiques globales sur le corpus ---")
    for k, v in arr.items():
        if len(v) > 0:
            print(f"{k}: moyenne={np.mean(v):.3f}, médiane={np.median(v):.3f}, min={np.min(v):.3f}, max={np.max(v):.3f}, std={np.std(v):.3f}")

    # --- Export CSV si demandé ---
    if args.csv:
        print("\n--- Export CSV et stats globales sur tout le corpus (mode batch spaCy) ---")
        out_path = get_save_path('lexical_stats_all.csv')
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, '') for k in ordered_keys})
        print(f"CSV exporté : {out_path}")

    # --- Graphe de cooccurrences (global) ---
    if args.cooc:
        print("\n--- Graphe de cooccurrences (global) ---")
        G = build_cooc_graph([t[1] for t in texts], stopwords=stopwords)
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        if args.min_edge_weight > 0:
            edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= args.min_edge_weight]
            G = G.edge_subgraph(edges_to_keep).copy()
        print(f"Nombre de noeuds: {G.number_of_nodes()}, arêtes: {G.number_of_edges()}")
        top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
        for u, v, d in top_edges:
            print(f"{u} -- {v} (poids: {d['weight']})")
        nx.write_gexf(G, get_save_path('cooc_graph_global.gexf'))
        print(f"Graphe global exporté pour Gephi : {get_save_path('cooc_graph_global.gexf')}")

    if args.graph:
        print("\n--- Visualisation et export du graphe global de cooccurrence ---")
        G = build_cooc_graph([t[1] for t in texts], stopwords=stopwords)
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        if args.min_edge_weight > 0:
            edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= args.min_edge_weight]
            G = G.edge_subgraph(edges_to_keep).copy()
        import matplotlib.pyplot as plt
        degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, d in degrees[:50]]
        H = G.subgraph(top_nodes)
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx(H, pos, with_labels=True, node_size=400, font_size=10)
        plt.title("Top 50 nœuds les plus connectés du sous-graphe géant")
        plt.tight_layout()
        plt.savefig(get_save_path('cooc_graph_top50.png'))
        print(f"Graphique sauvegardé : {get_save_path('cooc_graph_top50.png')}")
        nx.write_gexf(G, get_save_path('cooc_graph_global.gexf'))
        print(f"Graphe global exporté pour Gephi : {get_save_path('cooc_graph_global.gexf')}")
    return

if __name__ == "__main__":
    main()
