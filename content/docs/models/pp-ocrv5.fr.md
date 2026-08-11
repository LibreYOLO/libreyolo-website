---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5 : détection et reconnaissance de texte dans LibreYOLO'
description: >-
  Utilisez PP-OCRv5 dans LibreYOLO pour l'OCR multilingue de texte en situation
  réelle. Installez, prédisez et validez les checkpoints t et l sous licence
  Apache-2.0.
lead: >-
  PP-OCRv5 est le pipeline de détection et de reconnaissance de texte de
  PaddleOCR : un détecteur à binarisation différentiable localise les
  quadrilatères de texte et un module de reconnaissance SVTR/CTC les lit.
  LibreYOLO le porte vers PyTorch en deux niveaux.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - détection de texte
  - reconnaissance de texte
  - OCR scène réelle
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Quadrilatères
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRl-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Polygones (N, 4, 2) dans l'ordre de lecture : haut gauche, haut droit,

        # bas droit, bas gauche. Les quadrilatères de détection sont de vrais

        # polygones (texte pivoté), ils remplissent result.ocr, pas
        result.boxes.

        print(result.ocr.data.shape)

        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # métrique principale
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Installer

PP-OCRv5 ne nécessite aucun extra en plus du paquet de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

Chaque checkpoint réunit les deux étapes, la détection et la reconnaissance,
dans un seul fichier `.pt`, tandis que le jeu de caractères de reconnaissance et
les valeurs par défaut du pipeline figurent dans les métadonnées du checkpoint.
Le module de reconnaissance lit le chinois simplifié et traditionnel, l'anglais,
le japonais et le pinyin à l'aide d'un seul dictionnaire. `result.ocr` est une
charge utile `OCRRegions` : `.data` contient les polygones à quatre points,
`.texts` les transcriptions, `.conf` le score de reconnaissance de chaque région
et `.det_conf` le score de détection. Les sources contenant plusieurs images
s'exécutent séquentiellement : le pipeline à deux étapes ne forme pas de batch
entre les images. Consultez la [prédiction](/docs/predict) pour les sources, le
streaming et le traitement des résultats.

## Variantes

Deux niveaux sont proposés : `t`, construit sur les backbones plus légers
PP-LCNetV3/PP-OCRv5_mobile pour le CPU, et `l`, construit sur les backbones
serveur PP-HGNetV2 pour une meilleure exactitude. Les deux niveaux exécutent la
détection avec une limite fixe sur le côté long et reconnaissent les recadrages
par batch ; `rec_batch` contrôle le nombre de recadrages transmis au module de
reconnaissance à chaque passe.

## Valider

`val()` mesure le pipeline sur un dossier d'images accompagné d'un fichier
`labels/<split>.jsonl`, ou sur le YAML de dataset équivalent. Chaque étiquette
répertorie les polygones des régions de texte et leurs transcriptions pour
l'image correspondante. La méthode rapporte la moyenne harmonique de détection
(précision/rappel/F1 associés par IoU), le F1 de bout en bout (moyenne harmonique
plus correspondance exacte de la transcription après normalisation, la métrique
`fitness` du checkpoint) et 1-NED, la distance d'édition normalisée moyenne sur
les paires correspondantes.

<code-tabs name="val" />

## Exporter

<export-matrix />

PP-OCRv5 est un pipeline à deux réseaux, où détection et reconnaissance se
déplacent ensemble, et non un graphe unique pouvant être tracé. L'export n'est
pas implémenté : aucun format n'est encore pris en charge. Effectuez directement
le fine-tuning du code d'entraînement amont Apache-2.0 et convertissez le résultat
avec `weights/convert_ppocr_weights.py` si vous avez besoin d'un checkpoint hors
de ce format.

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
