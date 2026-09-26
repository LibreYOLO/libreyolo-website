---
title: Détection de points
seo_title: Détection de points et comptage dans LibreYOLO
description: >-
  Localiser les objets par des points uniques plutôt que des boîtes dans
  LibreYOLO. Prédire des centroïdes, compter les objets, entraîner FOMO et lire
  les métriques de points.
lead: >-
  La détection de points renvoie une position x, y par objet au lieu d'une boîte
  englobante. LibreYOLO l'expose comme la tâche point, et une prédiction
  contient une ligne x, y, classe et confiance par objet.
keywords:
  - détection points python
  - comptage objets python
  - détection centroïdes
  - localisation points FOMO
  - compter objets images
  - localisation par points
last_verified: 1.5.0
snippets:
  predict:
    - label: Prédire les points et les compter
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Les poids LibreFOMO ne sont pas téléchargés automatiquement. Récupérez

        # d'abord un checkpoint sur https://huggingface.co/LibreYOLO et
        chargez-le localement.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # nombre d'objets

        print(points.xy)       # centres (N, 2) en pixels de l'image d'origine

        print(points.cls, points.conf)
    - label: Coordonnées normalisées et nombres par classe
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # mêmes centres dans [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Entraîner FOMO sur un dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Prédire avec le checkpoint entraîné
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() recharge le meilleur checkpoint dans le même objet. À son
        # retour, le modèle prédit donc avec les poids entraînés.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # erreur moyenne de
        localisation

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # erreur de
        comptage
    - label: Modifier les seuils de distance
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Les bornes du balayage font partie du texte des clés. Un balayage

        # personnalisé renomme donc les clés mAP qu'il produit.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Définition

La tâche `point` localise chaque objet par une seule coordonnée x, y et une
classe, sans largeur, hauteur ni masque. Comme une prédiction est une liste
plate d'objets, le nombre de lignes correspond au nombre d'objets. C'est ce qui
en fait la tâche de comptage.

Une prédiction remplit `result.points`, une charge utile `Points` qui encapsule
un tableau `(N, 4)` de lignes `x, y, classe, confiance` exprimées en pixels de
l'image d'origine. `.xy` renvoie les coordonnées, `.xyn` les mêmes coordonnées
divisées par la taille de l'image, `.cls` les indices de classes et `.conf` les
scores. `len()` renvoie le nombre de points. `result.boxes` reste vide. Les
arguments `iou` et `max_det` n'ont donc aucun élément sur lequel agir.

## Modèles

Trois familles couvrent `point` et ne sont pas interchangeables.

[FOMO](/docs/models/fomo) constitue l'option à vocabulaire fixe. Ce
classificateur de grille étiquette chaque cellule d'une grille de basse
résolution comme arrière-plan ou comme centre d'objet. C'est la seule famille
de points que LibreYOLO peut entraîner et la seule qui s'exporte.

[LocateAnything](/docs/models/locate-anything) reçoit du texte plutôt qu'un
indice de classe. Son vocabulaire correspond donc à la phrase que vous écrivez.
Il nécessite l'extra `vlm`, se construit sous la forme `LibreLocateAnything`
plutôt que par la fabrique `LibreYOLO()`, et ses poids sont réservés à un usage
non commercial. Les conditions exactes, ainsi que les deux licences
supplémentaires que compose le checkpoint, figurent sur sa page.

[SenseNova-Vision](/docs/models/sensenova-vision) atteint `point` avec le même
checkpoint de génération guidée qu'il emploie pour six autres tâches, chargé
avec `LibreVLM("sensenova-vision", task="point")`. Il nécessite l'extra
`sensenova`, et chaque prédiction est une passe de génération sur un modèle 7B.
La latence par image est donc nettement plus élevée que celle d'un détecteur
spécialisé. Ses poids sont réservés à un usage non commercial. Leur licence
figure sur sa page.

## Prédire

Les poids LibreFOMO constituent l'unique exception au téléchargement automatique
sur ce site. `LibreYOLO("LibreFOMOs-point.pt")` recherche ce fichier sur le
disque et déclenche une `ValueError` qui le nomme au lieu de le récupérer.
Téléchargez d'abord un checkpoint depuis
l'[organisation LibreYOLO](https://huggingface.co/LibreYOLO) sur Hugging Face
et chargez-le par son chemin local, ou entraînez le vôtre.

<code-tabs name="predict" />

Le nom du fichier doit comporter le suffixe de tâche `-point` afin que le
chargeur le reconnaisse. `predict(..., nms_radius=1)` contrôle la distance en
cellules de grille nécessaire pour que deux détections FOMO soient toutes deux
conservées. Consultez la page [prédiction](/docs/predict) pour les sources, le
streaming et la gestion des résultats.

## Format du dataset

`point` ne possède aucun format d'annotation propre. Les familles de points
lisent la structure de détection YOLO standard et dérivent un centre de chaque
ligne de boîte. `cx cy` constitue le point, tandis que `w h` détermine seulement
si la ligne est valide.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Chaque fichier d'étiquettes contient une ligne par objet, avec des coordonnées
normalisées :

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Un fichier d'étiquettes absent ou vide signifie que l'image ne contient aucun
objet. Consultez les [formats de datasets](/docs/reference/dataset-formats)
pour le contrat complet.

## Entraîner

FOMO est la seule famille de points dotée d'une implémentation d'entraînement.
`train()` sur LocateAnything et SenseNova-Vision déclenche une
`NotImplementedError`. Effectuez leur fine-tuning dans le projet amont, puis
chargez le résultat.

<code-tabs name="train" />

`imgsz` n'est pas un choix libre pour FOMO. Sa valeur par défaut est la
résolution native du checkpoint chargé. Une valeur différente déclenche une
`ValueError` qui indique la taille attendue. Consultez la page
[entraînement](/docs/train) pour les datasets, les systèmes de journalisation et
le multi-GPU, ainsi que la [page FOMO](/docs/models/fomo) pour les valeurs par
défaut de cette famille.

## Valider

`val()` associe un à un les points prédits aux points de vérité terrain avec
l'algorithme hongrois, sur un balayage de seuils de distance. Un seuil est une
distance euclidienne dans les coordonnées normalisées de l'image. Le balayage
par défaut comporte dix valeurs de 0,01 à 0,10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` et `metrics/f1` sont des macro-moyennes
sur les classes au seuil le plus strict du balayage, soit 0,01 par défaut.
`metrics/mAP@0.01` est la précision moyenne à ce même seuil, et
`metrics/mAP@[0.01:0.10]` la moyenne sur l'ensemble du balayage. Cette dernière
sert également de `fitness`, la valeur lue pour sélectionner le meilleur
checkpoint. Les deux clés mAP sont construites à partir des seuils utilisés.
Transmettre `dist_thresholds=` les renomme donc.

`metrics/MLE` est la distance moyenne entre les paires associées au seuil le
plus strict, dans les mêmes unités normalisées. `metrics/MAE` et
`metrics/RMSE` sont des métriques de comptage plutôt que de localisation. Elles
mesurent l'écart par image entre le nombre de points prédits et celui des points
de vérité terrain.

FOMO ajoute par-dessus un second groupe à l'échelle de la grille. Il balaie la
confiance et `nms_radius`, puis publie la combinaison à meilleure F1 sous les
clés `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` et
`metrics/grid_FN`. Les réglages correspondants figurent sous
`decode/threshold` et `decode/nms_radius`.

## Exporter

FOMO s'exporte par le parcours partagé. Un artefact exporté se recharge par
`LibreYOLO()` grâce au suffixe de son fichier. Un fichier `.onnx` ou `.engine`
se comporte donc comme un checkpoint et renvoie le même `Results`.

<code-tabs name="export" />

La couverture de chaque format figure sur la [page FOMO](/docs/models/fomo) et
dans la [matrice d'exportation complète](/docs/reference/export-matrix).
LocateAnything et SenseNova-Vision ne s'exportent pas. `export()` déclenche une
erreur pour les deux, car un modèle génératif ne possède aucun graphe de
détection traçable.
