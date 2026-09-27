---
title: FOMO
families:
  - fomo
seo_title: "FOMO\_: localisation de points, entraîner et exporter dans LibreYOLO"
description: "Exécutez FOMO (Faster Objects, More Objects) dans LibreYOLO\_: un détecteur minuscule de localisation de points pour compter de nombreux petits objets. Installez, prédisez, entraînez et exportez."
lead: "FOMO est un localisateur de points fondé sur une grille\_: chaque cellule d'une grille basse résolution est classée comme fond ou comme centre d'objet, sans aucune régression de bounding box. LibreYOLO le prend en charge pour la tâche point."
keywords:
  - FOMO
  - Faster Objects More Objects
  - localisation de points
  - détection de centroïdes
  - détection de petits objets
  - IA embarquée edge
  - détection sur microcontrôleur
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Les poids LibreFOMO ne se téléchargent pas seuls (voir Checkpoints).
        # Visez ici un checkpoint déjà téléchargé en local.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz doit être passé : la CLI le met à 640 par défaut, et le

        # checkpoint s n'accepte que son 96 natif.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Installation

FOMO ne demande aucun extra au-delà du paquet de base.

```bash
pip install libreyolo
```

## Prédire

Contrairement à toutes les autres familles de ce site, les poids LibreFOMO ne
sont pas téléchargés automatiquement : `LibreYOLO("LibreFOMOs-point.pt")`
cherche ce fichier sur le disque et lève une `ValueError` qui le nomme, au lieu
d'aller le chercher sur Hugging Face. Téléchargez d'abord un checkpoint depuis
[l'organisation LibreYOLO](https://huggingface.co/LibreYOLO) et chargez-le par
son chemin local, ou entraînez le vôtre (voir Entraîner ci-dessous).

<code-tabs name="predict" />

Le résultat porte une charge utile `points` au lieu de `boxes` : chaque ligne
vaut `x, y, class, confidence`, disponible via `result.points.data` ou par les
accesseurs `.xy`, `.xyn`, `.cls` et `.conf`. Il n'y a aucun seuil `iou` à
régler, puisqu'il n'y a pas de boîtes à supprimer ; `predict(..., nms_radius=1)`
fixe le nombre de cellules de la grille qui doit séparer deux détections pour
qu'elles survivent toutes les deux, et le nom de fichier doit porter le suffixe
de tâche `-point` de FOMO pour que le chargeur le reconnaisse. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Trois tailles, `s`, `m` et `l`, emploient des backbones de style MobileNetV2 de
plus en plus larges, à des résolutions d'entrée fixes d'autant plus grandes,
chacun derrière une unique tête de classification 1x1. Cette famille ne porte
pas de table de benchmark ici ; la taille du fichier de checkpoint dans la table
ci-dessous est le signal par taille le plus clair publié à ce jour.

## Entraîner

<code-tabs name="train" />

`imgsz` n'est pas un choix libre : il prend par défaut la résolution native du
checkpoint chargé, et passer une autre valeur lève une `ValueError` qui nomme la
taille attendue. Ces tailles sont 96 pour `s`, 192 pour `m` et 224 pour `l`. La
CLI met `imgsz` à 640 par défaut, donc une commande `libreyolo train` doit la
fixer explicitement pour correspondre au checkpoint.

Laissé à ses réglages par défaut, l'entraîneur fait 40 époques avec un batch de
32 et Adam à `lr0=3e-4`, sans weight decay, et une classe d'avant-plan pondérée
100x par rapport au fond dans la loss d'entropie croisée par cellule, puisque
presque toutes les cellules de la grille sont du fond dans une scène typique.
L'EMA et la précision mixte sont toutes deux désactivées par défaut, et aucune
des augmentations géométriques ou colorimétriques employées ailleurs dans
LibreYOLO n'est appliquée : mosaic, mixup, jitter HSV, flip, rotation,
translation et cisaillement sont tous à zéro.

C'est le parcours avec lequel les checkpoints LibreFOMO publiés ont été
entraînés, à partir de zéro sur COCO.

Voir [l'entraînement](/docs/train) pour les datasets et les loggers.

## Valider

`val()` bascule vers un validateur au niveau de la grille conçu pour cette
famille. À côté des clés d'appariement de points `metrics/precision`,
`metrics/recall` et `metrics/mAP@` partagées avec les autres tâches point, il
balaie les seuils de confiance et les valeurs de `nms_radius`, puis publie la
combinaison au meilleur F1 sous `metrics/grid_F1`, `metrics/grid_precision`,
`metrics/grid_recall` et `metrics/grid_mean_distance`, ainsi que le seuil et le
rayon qui l'ont produite sous `decode/threshold` et `decode/nms_radius`.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. Exécuter le graphe dans un runtime nu,
sans LibreYOLO installé, est aussi pris en charge, mais le prétraitement et le
post-traitement sont alors à votre charge.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille. Aucun ne se télécharge
automatiquement : récupérez le fichier voulu sur la page Hugging Face liée et
passez son chemin local à `LibreYOLO()`.

<checkpoint-table />

## Licence

<provenance-box>

Il n'existe aucun dépôt de code amont vers lequel pointer pour FOMO : Edge
Impulse décrit la technique dans un billet de blog et dans la documentation de
son produit, mais n'a publié ni code d'entraînement ni code d'inférence pour
FOMO. L'architecture et l'entraînement présentés ici sont l'implémentation
propre à LibreYOLO de cette description publiée, et les checkpoints LibreFOMO
publiés sont entraînés à partir de zéro sur COCO, si bien que le code comme ces
poids sont sous licence MIT et appartiennent à LibreYOLO. Le nom FOMO et la
technique qu'il désigne restent ceux d'Edge Impulse.

</provenance-box>
