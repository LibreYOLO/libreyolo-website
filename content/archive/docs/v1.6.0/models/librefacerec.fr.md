---
title: LibreFaceRec
families:
  - facerec
seo_title: "LibreFaceRec\_: reconnaissance et vérification de visages"
description: "Utilisez LibreFaceRec dans LibreYOLO pour la détection de visages, l'embedding et la vérification. Installez et prédisez\_; les poids d'embedding sont sous Apache-2.0."
lead: "LibreFaceRec est la tâche d'embedding de visages de LibreYOLO\_: un détecteur de visages localise et aligne les visages, et une tête de reconnaissance produit un embedding d'identité normalisé L2 pour la vérification ou la recherche par similarité."
keywords:
  - LibreFaceRec
  - reconnaissance faciale python
  - embedding de visages
  - vérification de visages
  - comparer deux visages python
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Les noms librefacerec-* pointent vers cette famille quel que soit
        # le suffixe du fichier et se téléchargent au premier usage depuis
        # l'org Hugging Face de LibreYOLO, avec le détecteur par défaut.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), normalisés L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Vérifier
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Compare le visage le plus marquant de chaque image par similarité
        # cosinus de leurs embeddings normalisés L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Recherche dans une galerie
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # les visages de l'image
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # Similarités cosinus (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Installation

La tête de reconnaissance de LibreFaceRec passe par `onnxruntime`, qui ne fait
pas partie de l'installation de base.

```bash
pip install "libreyolo[onnx]"
```

## Prédire

<code-tabs name="predict" />

La détection et la reconnaissance sont deux graphes ONNX distincts derrière un
seul appel : un détecteur de visages localise et aligne chaque visage sur un
recadrage canonique, et la tête de reconnaissance renvoie un embedding
normalisé L2 par visage. Si vous n'y touchez pas, `predict()` télécharge et
associe automatiquement le détecteur par défaut fourni. `face_detector` accepte
un callable, un modèle de détection LibreYOLO ou une instance `FaceDetector` ;
`face_boxes` court-circuite entièrement la détection avec des boîtes dont vous
disposez déjà. `result.embeddings` contient une ligne par visage détecté,
alignée sur `result.boxes` ; sa méthode `.similarity()` calcule la similarité
cosinus face à un autre embedding ou à une galerie entière en un seul appel.
Pour comparer directement deux images plutôt que deux embeddings déjà calculés,
`model.verify(image_a, image_b)` lance la détection et l'embedding sur les deux
et compare leur visage au score de confiance le plus élevé. Tout autre modèle
de reconnaissance ONNX suivant la convention ArcFace (recadrage aligné en
entrée, embeddings `(N, D)` en sortie) peut être substitué en passant son
chemin de fichier au lieu d'un nom `librefacerec-*`. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Exporter

<export-matrix />

LibreFaceRec encapsule déjà un graphe ONNX pré-exporté ; le réexporter vers un
autre format n'est pas implémenté.

## Licence

<provenance-box>

Le détecteur de visages par défaut fourni est un second artefact sous une
seconde licence : YuNet d'OpenCV Zoo, MIT, copyright Shiqi Yu. Aucun code
d'architecture n'est porté depuis l'un ou l'autre projet ; les deux graphes
sont consommés de façon opaque via `onnxruntime`, si bien que le wrapper propre
à LibreYOLO ne contient aucun code tiers et est MIT de bout en bout.

</provenance-box>

## Citation

<citation-block />
