"""
normaliz.py — Traitement complet d'un fichier ZDTR brut :
  1. Nettoyage  : supprime en-têtes, séparateurs, totaux ; fusionne les lignes continues
  2. Structuration : extrait les champs et produit le CSV utilisé par app.py
"""
import re
import csv

# ── Fichiers ────────────────────────────────────────────────────────────────
INPUT_RAW   = "_01.07.2026_ZDTR.txt"
OUTPUT_CLEAN = "_01.07.2026_ZDTR_clean.txt"
OUTPUT_CSV   = "_01.07.2026_ZDTR_structured.csv"

# ── Patterns de détection des lignes ────────────────────────────────────────
DATA_RE = re.compile(r"^\d{4,5}\s+")                        # ligne de donnée (code produit)
SEP_RE  = re.compile(r"^\s*-{10,}\s*$")                     # ligne séparateur "----------"
DUR_RE  = re.compile(r"^\s+(\d+\s+h\s+\d+\s+m|\d+\s*m)\s*$")  # durée seule (sous-total)

# Les lignes de continuation (description qui déborde) ont 81-83 espaces d'indentation
CONTINUATION_MIN = 75

# ── PHASE 1 : nettoyage ─────────────────────────────────────────────────────
with open(INPUT_RAW, "rb") as f:
    raw = f.read()
text = raw.decode("latin-1")
lines = text.splitlines()

clean_lines = []
current = None   # ligne de donnée en cours de construction
after_sep = False

for i, line in enumerate(lines):
    if i < 6:          # 6 lignes d'en-tête rapport (Bimbo, Downtime, From…)
        continue

    stripped = line.strip()

    # Ligne vide → enregistre la ligne courante
    if not stripped:
        if current is not None:
            clean_lines.append(current)
            current = None
        after_sep = False
        continue

    # Séparateur "----------"
    if SEP_RE.match(line):
        after_sep = True
        continue

    # Durée seule après séparateur → l'ajouter à la description
    if after_sep and DUR_RE.match(line):
        if current is not None:
            current = current + " " + stripped
        after_sep = False
        continue

    # Ligne de donnée (commence par un code 4-5 chiffres)
    if DATA_RE.match(line):
        if current is not None:
            clean_lines.append(current)
        current = stripped
        after_sep = False
        continue

    # Ligne de continuation de description (≥ 75 espaces d'indentation, texte réel)
    indent = len(line) - len(line.lstrip())
    if indent >= CONTINUATION_MIN and current is not None and not DUR_RE.match(line):
        current = current + " " + stripped
        after_sep = False
        continue

    # Tout le reste : en-têtes de section, sous-totaux machine, grand total → ignore
    after_sep = False

if current is not None:   # dernière ligne non suivie d'une ligne vide
    clean_lines.append(current)

with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f_out:
    f_out.write("\n".join(clean_lines) + "\n")
print(f"✅ Nettoyage  : {len(clean_lines):,} lignes → {OUTPUT_CLEAN}")

# ── PHASE 2 : structuration CSV ─────────────────────────────────────────────
# Durée acceptée : "09 m", "00 m", "1 h 20 m", "6 h 30 m", etc.
CSV_RE = re.compile(
    r"^(?P<code>\d{4,5})\s+(?P<produit>.*?)\s*(?P<type>MECHANICAL|ELECTRICAL|OTHER)\s+"
    r"(?P<date>\d{2}\.\d{2}\.\d{4})\s+(?P<duree>\d+\s*(?:h\s+\d+\s*)?m)\s+(?P<desc>.*)$"
)

rows = []
skipped = 0
for line in clean_lines:
    m = CSV_RE.match(line)
    if m:
        rows.append(m.groupdict())
    else:
        skipped += 1

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(
        f_out,
        fieldnames=["code", "produit", "type", "date", "duree", "desc"],
        delimiter=";",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Structuration : {len(rows):,} lignes extraites, {skipped} ignorées → {OUTPUT_CSV}")
if skipped:
    print("   (lignes ignorées = en-têtes résiduels ou formats non reconnus)")
