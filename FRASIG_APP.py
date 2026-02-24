import os
import warnings
# Tvinga protobuf att använda python-implementationen för bättre kompatibilitet
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, render_template_string, request
import spacy
import benepar
import nltk
from nltk.tree import Tree

# Inställningar
warnings.filterwarnings("ignore")
app = Flask(__name__)

# --- FIX FÖR RENDER: Ladda ner Benepar-modell till lokal mapp ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

print("Laddar High-Precision-modeller...")

# 1. Ladda ner Benepar-modellen om den saknas
try:
    benepar.download('benepar_sv2', download_dir=nltk_data_path)
except Exception as e:
    print(f"Information om modellnedladdning: {e}")

# 2. Ladda spaCy-modellen (Medium-versionen för att spara RAM)
nlp = spacy.load("sv_core_news_md")

# 3. Koppla ihop Benepar med spaCy
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_sv2"})

# Översättningskarta för etiketter
label_map = {
    "S": "Sats",
    "NP": "NP",
    "VP": "VP",
    "PP": "PP",
    "ADJP": "AdjP",
    "AP"  : "AdjP",
    "ADVP": "AdvP",
    "AVP": "AdvP",
    "PRN": "Parentes",
    "NN": "Subst",
    "PM": "Egennamn",
    "VB": "Verb",
    "JJ": "Adj",
    "AB": "Adv",
    "PN": "Pron",
    "PS": "Poss. pron.",
    "HP": "Rel. pron.",
    "HA": "Rel. adv.",
    "P" : "Prep",
    "KN": "Konj",
    "SN": "Subj",
    "IE": "Inf-märke",
    "DT": "Determinerare",
    "RG": "Räkn Grundtal",
    "RO": "Räkn Ordningstal",
    "PC": "Particip",
    "PL": "Partikel",
    "UO": "Utl",
    "INTJ": "Interj",
    "MAD": "Skilj",
    "MID": "Skilj",
    "PAD": "Par",
}

def clean_and_translate_tree(tree):
    """
    Rensar XP-noder, översätter etiketter och sätter ensamma verb i VP.
    """
    if isinstance(tree, str):
        return tree

    new_children = []
    for child in tree:
        if isinstance(child, Tree):
            # Lyft upp barn från XP-noder
            if child.label() == "XP":
                new_children.extend([clean_and_translate_tree(c) for c in child])
            
            # Skapa VP för ensamma verb som inte redan ligger i en VP
            elif child.label() == "VB" and tree.label() != "VP":
                v_node = clean_and_translate_tree(child)
                new_vp = Tree("VP", [v_node])
                new_children.append(new_vp)
            
            else:
                new_children.append(clean_and_translate_tree(child))
        else:
            new_children.append(child)
    
    tree.clear()
    tree.extend(new_children)

    # Översätt etiketten
    label = tree.label()
    tree.set_label(label_map.get(label, label))
    
    return tree

# HTML-mall för webbsidan
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>FRASIG - Interaktiv PSG</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; display: flex; justify-content: center; padding: 40px; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); width: 95%; text-align: center; }
        h1 { color: #1a73e8; }
        input[type="text"] { width: 70%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
        button { padding: 12px 25px; background: #1a73e8; color: white; border: none; border-radius: 8px; cursor: pointer; }
        .result-area { margin-top: 40px; border-top: 1px solid #eee; overflow-x: auto; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        svg { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>FRASIG</h1>
        <form method="POST">
            <input type="text" name="sentence" placeholder="Skriv en mening..." value="{{ sentence }}" required>
            <button type="submit">Analysera</button>
        </form>
        {% if tree_svg %}
        <div class="result-area">
            <h3>Resultat för: <i>"{{ sentence }}"</i></h3>
            {{ tree_svg|safe }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    tree_svg = None
    sentence = ""
    if request.method == "POST":
        sentence = request.form.get("sentence", "")
        try:
            doc = nlp(sentence)
            sent = list(doc.sents)[0]
            psg_tree = Tree.fromstring(sent._.parse_string)
            
            # Kör städning och översättning
            psg_tree = clean_and_translate_tree(psg_tree)
            
            tree_svg = psg_tree._repr_svg_()
        except Exception as e:
            print(f"Fel vid analys: {e}")
            
    return render_template_string(HTML_TEMPLATE, tree_svg=tree_svg, sentence=sentence)

if __name__ == "__main__":
    # Debug-utskrift för att skriva ut porten
    port = int(os.environ.get("PORT", 5000))
    print(f"Using port: {port}")  # Debug-utskrift av porten
    app.run(host="0.0.0.0", port=port)
