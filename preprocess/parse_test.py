import xml.etree.ElementTree as ET

def parse_test_parti(path_test, path_ref):
    """
    Parse le corpus XML de test et retourne un dictionnaire :
    { parti: [discours, discours, ...] }
    """

    # --- Création de listes vides pour chaque parti politique

    parties = {
        "ELDR": [],
        "GUE-NGL": [],
        "PPE-DE": [],
        "PSE": [],
        "Verts-ALE": []
    }

    # --- Lecture du fichier txt (id -> parti)

    id_to_party = {}

    with open(path_ref, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # On ignore les lignes vides
            if not line:
                continue

            # On vérifie qu'il y a bien 2 colonnes
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            doc_id, party = parts
            id_to_party[doc_id] = party

    # --- Parsing du XML ---

    tree = ET.parse(path_test)
    root = tree.getroot()

    for doc in root.findall("doc"):
        doc_id = doc.get("id")

        # On garde seulement les docs présents dans le fichier de référence
        if doc_id not in id_to_party:
            continue

        party = id_to_party[doc_id]

        # On vérifie que le parti est dans notre mapping
        if party not in parties:
            continue

        # On extrait tout le texte des balises <p> à l'intérieur des balises <doc>
        speech = " ".join(
            " ".join(p.itertext()).strip()
            for p in doc.findall(".//p")
            if "".join(p.itertext()).strip()
        )

        if speech:
            parties[party].append(speech)

    return parties
