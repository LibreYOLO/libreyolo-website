---
title: OCR
seo_title: 'OCR : détection et reconnaissance de texte dans LibreYOLO'
description: >-
  Trouver et lire le texte des images avec LibreYOLO. Prédire des quadrilatères
  et des transcriptions, annoter un dataset JSONL et valider avec la moyenne
  harmonique, la F1 de bout en bout et 1-NED.
lead: >-
  L'OCR localise le texte dans une image et le lit. LibreYOLO l'expose comme la
  tâche ocr, qui renvoie un polygone à quatre points et une transcription par
  région de texte, dans l'ordre de lecture.
keywords:
  - bibliothèque ocr python
  - reconnaissance texte scène
  - détection texte quadrilatères
  - PP-OCRv5 python
  - détection lecture texte bout en bout
last_verified: 1.5.0
snippets:
  predict:
    - label: Lire le texte d'une image
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Le niveau t est le plus léger des deux et vise le CPU. SAMPLE_IMAGE

        # garde cet exemple exécutable ; remplacez-le par votre propre image
        textuelle.

        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(len(regions), "regions")

        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Lire les quadrilatères
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        regions = result.ocr

        print(regions.data.shape)   # polygones (N, 4, 2), HG HD BD BG

        print(regions.xyxy)         # enveloppes alignées sur les axes de ces
        polygones

        print(regions.det_conf)     # score de détection, distinct de .conf
    - label: Filtrer par confiance de reconnaissance
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRt-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Indexez avec des positions, pas avec un masque booléen : le découpage

        # conserve les transcriptions et les deux tableaux de scores avec la
        géométrie.

        regions = result.ocr.numpy()

        keep = regions[np.flatnonzero(regions.conf >= 0.9)]

        print(keep.texts)
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Définition

La tâche `ocr` effectue deux opérations en un seul appel : elle localise toutes
les régions de texte d'une image et les transcrit. Les régions sont renvoyées
sous forme de polygones à quatre points plutôt que de boîtes alignées sur les
axes, car le texte des scènes est souvent incliné. Elles sont ordonnées du haut
vers le bas, puis de gauche à droite.

Une prédiction remplit `result.ocr` avec une charge utile `OCRRegions`. `.data`
est un tableau flottant `(N, 4, 2)` de polygones exprimés en pixels de l'image
d'origine, ordonnés coin supérieur gauche, supérieur droit, inférieur droit,
inférieur gauche. `.texts` est la liste des N transcriptions. `.conf` est le
score de reconnaissance par région, et `.det_conf` le score de détection.
`.xyxy` fournit l'enveloppe alignée sur les axes de chaque polygone. Comme les
quadrilatères sont de véritables polygones, ils ne remplissent pas
`result.boxes`. Découper un `OCRRegions` conserve les transcriptions et les deux
tableaux de scores avec la géométrie.

## Modèles

Deux familles couvrent la tâche `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) est le pipeline dédié. Un détecteur à
binarisation différentiable trouve les quadrilatères de texte, puis un système
de reconnaissance SVTR/CTC les lit. Les deux étapes sont regroupées dans un
seul fichier `.pt` avec le jeu de caractères de reconnaissance. Il est publié
en deux niveaux, une variante légère pour le CPU et une variante serveur plus
précise. Un seul dictionnaire couvre le chinois simplifié et traditionnel,
l'anglais, le japonais et le pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) effectue l'OCR en générant les
mots sous forme de texte balisé depuis le même checkpoint 7B qui couvre ses six
autres tâches, chargé avec
`LibreVLM("sensenova-vision", task="ocr")`. Il nécessite l'extra
`sensenova` et ses poids sont réservés à un usage non commercial. Sa page
précise leur licence.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

PP-OCRv5 effectue la détection avec une limite fixe sur le côté le plus long,
puis reconnaît les régions recadrées par lots. `rec_batch` contrôle le nombre
de recadrages transmis au système de reconnaissance à chaque propagation. Les
sources comportant plusieurs images sont traitées séquentiellement, car un
pipeline à deux étapes ne regroupe pas les images en lots. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

Les annotations OCR sont stockées dans un fichier JSONL par partition, avec un
objet JSON par image, à côté des images elles-mêmes.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Chaque ligne nomme une image et répertorie ses régions :

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` est un quadrilatère à quatre points en coordonnées de pixels
absolues, ordonnés coin supérieur gauche, supérieur droit, inférieur droit,
inférieur gauche. Une région dont le texte est illisible reçoit l'annotation
`"text": "###"`, convention ICDAR pour les zones à ignorer. Elle est exclue du
score de reconnaissance, et une prédiction qui la chevauche est ignorée au lieu
d'être comptée comme un faux positif.

Il suffit de transmettre le répertoire racine dans `data=`. Un fichier YAML de
dataset constitue l'autre possibilité, avec `path`, les noms de répertoires
facultatifs `images` et `labels`, ainsi que `nc: 1` et `names: {0: text}` comme
valeurs réservées du schéma, puisqu'un modèle OCR renvoie `Results.ocr` plutôt
que des détections. Consultez les
[formats de datasets](/docs/reference/dataset-formats) pour le contrat complet.

## Entraîner

Aucune des deux familles OCR ne possède d'implémentation d'entraînement.
`train()` déclenche une `NotImplementedError` dans les deux cas, et la prise en
charge de l'OCR se limite à la prédiction et à la validation. La page PP-OCRv5
indique le code d'entraînement amont sous licence Apache-2.0 ainsi que le script
de conversion qui réimporte dans LibreYOLO un checkpoint affiné.

## Valider

`val()` évalue l'ensemble du pipeline, en réunissant la détection et la
reconnaissance. Les polygones prédits sont associés un à un aux polygones de
vérité terrain lorsque leur IoU dépasse 0,5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` et `metrics/det_hmean` évaluent
uniquement la localisation. Une correspondance exige seulement le
chevauchement du polygone, quel que soit le contenu de la transcription.
`metrics/e2e_precision`, `metrics/e2e_recall` et `metrics/e2e_f1` ajoutent la
lecture. Une correspondance exige alors le même chevauchement du polygone et
une transcription exactement identique après normalisation NFKC et suppression
des espaces. La comparaison reste sensible à la casse. `metrics/e2e_f1` sert
également de `fitness`, la valeur lue pour sélectionner le meilleur checkpoint.

`metrics/rec_1-NED` évalue séparément le système de reconnaissance sur les
paires déjà associées par la détection. Il s'agit de un moins la distance
d'édition normalisée. Une transcription qui diffère d'un caractère obtient
ainsi un score proche de 1, tandis que la F1 de bout en bout lui attribue 0.

## Exporter

Aucun format d'exportation n'est disponible pour cette tâche. PP-OCRv5 réunit
deux réseaux qui se déplacent ensemble plutôt qu'un seul graphe traçable, et
`export()` déclenche une erreur pour chaque format dans les deux familles. Pour
déployer hors de LibreYOLO, effectuez le fine-tuning dans le projet amont et
utilisez son parcours de déploiement.
