import xml.etree.ElementTree as ET

def parse_train_parti(path_train):
    """
    Parse le fichier XML de train et retourne un dictionnaire :
    { parti: [discours, discours, ...] }
    """
    # --- Création de listes vides pour chaque parti politique ---

    parties = {
        "ELDR": [],
        "GUE-NGL": [],
        "PPE-DE": [],
        "PSE": [],
        "Verts-ALE": []
    }

    # --- Parsing du XML ---

    tree = ET.parse(path_train)
    root = tree.getroot()

    for doc in root.findall(".//doc"):
        parti_elem = doc.find(".//PARTI")
        if parti_elem is None:
            continue

        parti = parti_elem.get("valeur")

        paragraphs = doc.findall(".//texte//p")
        texte = " ".join(p.text.strip() for p in paragraphs if p.text)

        if parti in parties and texte:
            parties[parti].append(texte)

    return parties

