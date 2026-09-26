---
title: SSD
families:
  - ssd
seo_title: "SSD (SSD300)\_: détection d'objets dans LibreYOLO"
description: "Exécutez SSD300 dans LibreYOLO\_: un détecteur VGG16 en une seule passe pour la prédiction, la validation et l'export ONNX sous licence BSD-3-Clause. Aucun entraînement."
lead: >-
  SSD (Single Shot MultiBox Detector) prédit toutes les bounding boxes et tous
  les scores de classes depuis une grille dense de bounding boxes par défaut, en
  une seule passe forward et sans étape distincte de proposition de régions.
  LibreYOLO fournit le checkpoint SSD300 basé sur VGG16 comme détecteur réservé
  à l'inférence.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - détection d'objets
  - VGG16
  - détecteur avec ancres
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # imgsz est volontairement omis ici : SSD300 est tracé sur le canevas
        natif

        # de son checkpoint et toute autre valeur lève une erreur avant le début
        de l'export.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreSSD300.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Installer

SSD ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui de toutes les familles, le remplacement du
détecteur ne demande donc de modifier qu'une ligne. SSD décode sa grille de
bounding boxes par défaut avec des scores par classe, puis exécute la
suppression non maximale. `conf`, `iou` et `max_det` ont ainsi tous un effet
réel ici, contrairement aux détecteurs basés sur des requêtes de cette
bibliothèque. Consultez la [prédiction](/docs/predict) pour les sources, le
streaming et la gestion des résultats.

## Variantes

SSD fournit un seul checkpoint\u00a0: le réseau SSD300 basé sur VGG16, avec son
canevas natif fixe. Cette famille ne propose aucun choix de taille ou
d'échelle\u00a0; la prédiction, la validation et l'export utilisent tous ce même
graphe.

Le fichier de poids est `LibreSSD300.pt`, soit le préfixe de la famille suivi
de son unique clé de taille, `"300"`. La classe correspondante est `LibreSSD`,
une construction directe s'écrit donc `LibreSSD(size="300")` plutôt qu'avec
une classe portant le nom du fichier.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset au format utilisé
pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

SSD s'exporte uniquement vers ONNX\u00a0; tous les autres formats sont actuellement
bloqués pour cette famille. L'export utilise toujours le canevas natif du
checkpoint et le graphe expose la tête brute compactée de SSD plutôt qu'une
sortie avec suppression non maximale fusionnée. `nms=True` n'est donc pas
accepté au moment de l'export. Les backends propres à LibreYOLO exécutent les
étapes de décodage et de suppression après le rechargement du graphe.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Le code SSD300 de LibreYOLO n'est pas porté depuis la version Caffe des
auteurs de l'article\u00a0; il dérive de l'implémentation SSD300 sous licence
BSD-3-Clause de torchvision, dont le dépôt est indiqué ci-dessus comme source
upstream. Les poids VGG16 du backbone remontent eux-mêmes au VGGNet réduit et
entièrement convolutionnel d'Oxford, publié sous licence CC BY 4.0 par Karen
Simonyan et Andrew Zisserman.

</provenance-box>

## Citation

<citation-block />

