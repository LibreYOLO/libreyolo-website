---
title: DexiNed
families:
  - dexined
seo_title: "DexiNed\_: détection de contours, avec votre propre checkpoint"
description: >-
  Utilisez DexiNed dans LibreYOLO pour prédire une carte dense de probabilité de
  contours. Convertissez un checkpoint dont vous avez la licence, puis prédisez,
  validez et exportez-le.
lead: "DexiNed (Dense Extreme Inception Network) est un réseau convolutif qui prédit une carte dense de probabilité de contours à partir d'une seule image RGB. LibreYOLO enveloppe son architecture pour la détection de contours uniquement\_; aucun checkpoint n'est livré avec la bibliothèque."
keywords:
  - DexiNed
  - détection de contours python
  - edge detection deep learning
  - Dense Extreme Inception Network
  - BIPED
  - carte de contours
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 dans [0, 1]
        print(edges.binary(0.5).sum())  # pixels de contour après seuillage
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreDexiNedb-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # F-mesure à l'échelle optimale du
        dataset

        print(metrics["metrics/OIS"])   # F-mesure à l'échelle optimale de
        l'image
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx
        imgsz=352

        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: 342597fde3c4ba65
---

## Installation

DexiNed ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

LibreYOLO ne fournit aucun checkpoint DexiNed. Les poids officiellement publiés
sont entraînés sur BIPED, dont les conditions d'utilisation publiées limitent
l'usage à des fins non commerciales, si bien que LibreYOLO ne les met pas en
miroir. Convertissez un checkpoint que vous avez le droit d'utiliser avec
`weights/convert_dexined_weights.py`, qui vérifie les clés des tenseurs face à
l'architecture du runtime avant d'écrire un fichier que LibreYOLO peut charger
directement :

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` contient le résultat : un tableau float32 `(H, W)` dans
`[0, 1]`, avec `.binary(threshold)` qui renvoie un masque de contours booléen.
Il n'y a pas de boîtes, donc `conf`, `iou` et `max_det` n'ont aucun effet. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

DexiNed est fourni en une seule taille dans LibreYOLO. Le harnais de benchmark
de LibreYOLO n'a pas mesuré cette famille, il n'y a donc aucun chiffre publié
auquel la comparer.

## Valider

`val()` renvoie les F-mesures ODS et OIS de style BSDS face à un dataset de
contours apparié : des images à côté de cartes de contours de même nom de base,
avec un masque de validité optionnel pour que les pixels de padding ne comptent
jamais. `imgsz` doit être divisible par le pas de sous-échantillonnage du
réseau, et LibreYOLO lève une erreur claire si ce n'est pas le cas.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export de contours suit un contrat de runtime à résolution fixe et à batch 1 :
`dynamic` et un `batch` autre que 1 sont refusés, et le graphe exporté produit
une seule carte de probabilité fusionnée. Un artefact exporté se recharge via
`LibreYOLO()` d'après l'extension de son fichier, si bien qu'un fichier `.onnx`
se comporte comme un checkpoint et renvoie le même `Results`.

<code-tabs name="export" />

## Licence

<provenance-box>

LibreYOLO ne publie aucun checkpoint DexiNed. Rien n'est mis en miroir sous
l'organisation LibreYOLO ; convertissez plutôt, avec
`weights/convert_dexined_weights.py`, un checkpoint pour lequel vous détenez une
licence.

</provenance-box>

## Citation

<citation-block />
