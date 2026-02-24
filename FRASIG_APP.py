# Lägg till detta högst upp vid de andra import-raderna
import benepar
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, render_template_string, request
import spacy
from benepar import BeneparComponent
from nltk.tree import Tree
import warnings

# Inställningar
warnings.filterwarnings("ignore")
app = Flask(__name__)

# 1. Ladda modellerna vid start
print("Laddar High-Precision-modeller...")
# Lägg till detta precis innan nlp = spacy.load...
try:
    benepar.download_model("benepar_sv2")
except Exception as e:
    print(f"Modellen fanns redan eller kunde inte laddas: {e}")
nlp = spacy.load("sv_core_news_md")
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_sv2"})

label_map = {
    # Fraser
    "S": "Sats",
    "NP": "NP",
    "VP": "VP",
    "PP": "PP",
    "ADJP": "AdjP",
    "AP"  : "AdjP",
    "ADVP": "AdvP",
    "AVP": "AdvP",
    "PRN": "Parentes",

    # Ordklasser
    "NN": "Subst",
    "PM": "Egennamn",  # VIKTIG: För namn på personer/platser
    "VB": "Verb",
    "JJ": "Adj",
    "AB": "Adv",
    "PN": "Pron",
    "PS": "Poss. pron.",
    "HP": "Rel. pron.",   # Relativa pronomen ("som")
    "HA": "Rel. adv.",     # Relativa adverb ("där")
    "P" : "Prep",
    "KN": "Konj",
    "SN": "Subj",
    "IE": "Inf-märke",    # Ordet "att"
    "DT": "Determinerare",
    "RG": "Räkn Grundtal",
    "RO": "Räkn Ordningstal",
    "PC": "Particip",
    "PL": "Partikel",
    "UO": "Utl",
    "INTJ": "Interj",

    # Skiljetecken
    "MAD": "Skilj",
    "MID": "Skilj",
    "PAD": "Par",
}

def clean_and_translate_tree(tree):
    """
    Rensar XP, översätter etiketter och sätter ensamma finita verb i egna VP.
    """
    if isinstance(tree, str):
        return tree

    new_children = []
    for child in tree:
        if isinstance(child, Tree):
            # 1. Eliminera XP-nivån (lyft upp barnen)
            if child.label() == "XP":
                new_children.extend([clean_and_translate_tree(c) for c in child])
            
            # 2. Skapa VP för ensamma verb (t.ex. "började")
            # Om barnet är ett Verb (VB) men dess förälder INTE är en VP
            elif child.label() == "VB" and tree.label() != "VP":
                # Skapa en ny VP-nod och lägg verbet inuti den
                v_node = clean_and_translate_tree(child)
                new_vp = Tree("VP", [v_node])
                new_children.append(new_vp)
            
            else:
                new_children.append(clean_and_translate_tree(child))
        else:
            new_children.append(child)
    
    tree.clear()
    tree.extend(new_children)

    # 3. Översätt etiketter från label_map
    label = tree.label()
    tree.set_label(label_map.get(label, label))
    
    return tree

# 2. HTML-mall
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
            
            # Kör städningen (XP-lyften)
            psg_tree = clean_and_translate_tree(psg_tree)
            
            tree_svg = psg_tree._repr_svg_()
        except Exception as e:
            print(f"Fel: {e}")
            
    return render_template_string(HTML_TEMPLATE, tree_svg=tree_svg, sentence=sentence)

if __name__ == "__main__":

    app.run(debug=False, port=5000)

