import json
from datetime import datetime

# Chemin du fichier à modifier
input_path = r"c:\Users\nicol\Documents\Projects\newspapers-analysis\data\processed\articles_v1.json"
output_path = r"c:\Users\nicol\Documents\Projects\newspapers-analysis\data\processed\articles_v1_filtered.json"  # Change en input_path pour écraser

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