---
title: TEED
families:
  - teed
seo_title: "TEED\_: détection de contours avec votre propre checkpoint"
description: >-
  Utilisez TEED dans LibreYOLO pour prédire une carte dense de probabilité des
  contours. Convertissez un checkpoint sous licence, puis prédisez, validez et
  exportez-le.
lead: "TEED (Tiny and Efficient Edge Detector) est un petit réseau convolutionnel qui prédit une carte dense de probabilité des contours à partir d'une image RGB. LibreYOLO encapsule son architecture uniquement pour la détection de contours\_; aucun checkpoint n'est fourni avec la bibliothèque."
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - détection de contours
  - BIPED
  - prédiction dense
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 dans [0, 1]

        print(edges.binary(0.5).sum())  # nombre de pixels de contour après
        seuillage
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # mesure F optimale à l'échelle du
        dataset

        print(metrics["metrics/OIS"])   # mesure F optimale à l'échelle de
        l'image
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Installer

TEED ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

LibreYOLO ne fournit aucun checkpoint TEED. Les poids publiés officiellement
sont entraînés sur BIPED, dont les conditions publiées limitent l'utilisation
à des fins non commerciales. LibreYOLO ne les réplique donc pas. Convertissez
un checkpoint que vous êtes autorisé à utiliser avec
`weights/convert_teed_weights.py`, qui vérifie les clés des tenseurs par
rapport à l'architecture du runtime avant d'écrire un fichier directement
chargeable par LibreYOLO\u00a0:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` contient le résultat\u00a0: un tableau float32 `(H, W)` dans
`[0, 1]`, dont `.binary(threshold)` renvoie un masque booléen des contours. Il
n'y a aucune bounding box, donc `conf`, `iou` et `max_det` n'ont aucun effet.
Consultez la [prédiction](/docs/predict) pour les sources, le streaming et la
gestion des résultats.

## Variantes

LibreYOLO fournit une seule taille de TEED. Le banc de benchmark de LibreYOLO
n'a pas mesuré cette famille, aucune valeur publiée ne permet donc de la
comparer aux autres.

## Valider

`val()` rapporte les mesures F ODS et OIS de type BSDS sur un dataset de
contours apparié\u00a0: les images sont placées à côté de cartes de contours de
même nom de base, avec un masque de validité facultatif pour que les pixels de
remplissage ne soient jamais comptés. `imgsz` doit être divisible par le
stride de sous-échantillonnage du réseau. LibreYOLO lève une erreur explicite
si ce n'est pas le cas.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export des contours suit un contrat de runtime à résolution fixe et avec un
batch de 1\u00a0: `dynamic` et toute valeur de `batch` autre que 1 sont rejetés, et
le graphe exporté produit une seule carte de probabilités fusionnée. Un
artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` se comporte donc comme un checkpoint et renvoie
les mêmes `Results`.

<code-tabs name="export" />

## Licence

<provenance-box>

LibreYOLO ne publie aucun checkpoint TEED. Rien n'est répliqué sous
l'organisation LibreYOLO\u00a0; convertissez plutôt un checkpoint pour lequel vous
possédez une licence avec `weights/convert_teed_weights.py`.

</provenance-box>

## Citation

<citation-block />

