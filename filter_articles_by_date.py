import json
import os
from datetime import datetime
from pathlib import Path
from src.utils.config_loader import load_config

# Charger la configuration
config_path = os.path.join(Path(__file__).parent, 'config', 'config.yaml')
config = load_config(config_path)

# Récupérer les chemins depuis la configuration
processed_dir = config['data']['processed_dir']
input_path = os.path.join(processed_dir, 'articles_v1.json')
output_path = os.path.join(processed_dir, 'articles_v1_filtered.json')

# Date limite
limit_date = datetime(1950, 1, 1)

with open(input_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

def keep_article(article):
    date_str = article.get("date", "")
    try:
        article_date = datetime.strptime(date_str, "%Y-%m-%d")
        return article_date >= limit_date
    except Exception:
        # Garde les articles sans date ou mal formés (optionnel : tu peux choisir de les exclure)
        return False

filtered = [a for a in articles if keep_article(a)]

print(f"Articles initiaux : {len(articles)}")
print(f"Articles après 1950 : {len(filtered)}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Fichier filtré sauvegardé sous : {output_path}")