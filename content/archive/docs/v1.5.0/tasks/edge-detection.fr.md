---
title: Détection des contours
seo_title: Détection des contours dans LibreYOLO
description: >-
  Prédire une carte dense de probabilité des contours depuis une image dans
  LibreYOLO. Convertir un checkpoint, appliquer un seuil à la carte, valider
  avec ODS et OIS, puis exporter.
lead: >-
  La détection des contours prédit la probabilité que chaque pixel se trouve sur
  la limite d'un objet. LibreYOLO l'expose comme la tâche edge, qui renvoie une
  carte dense de probabilités sur le canevas de l'image d'origine plutôt qu'un
  ensemble de segments.
keywords:
  - détection contours python
  - détection limites deep learning
  - carte probabilité contours
  - mesure F ODS OIS
  - prédiction dense contours
last_verified: 1.5.0
snippets:
  predict:
    - label: Prédire une carte de contours
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreYOLO ne fournit aucun checkpoint de contours ; convertissez-en
        d'abord un.

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # float32 (H, W) dans [0, 1]

        print(edges.binary(0.5).sum())    # nombre de pixels de contour au seuil
        0.5
    - label: Choisir votre propre seuil
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # La carte continue est conservée afin que le choix du seuil vous
        appartienne.

        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Enregistrer la visualisation
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # plot() rend la carte ; cette méthode est définie pour les contours et
        les normales.

        result.plot().save("edges.png")
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Modifier le balayage et la tolérance de correspondance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Définition

La tâche `edge` prédit une probabilité par pixel depuis une image RVB unique :
`0` signifie l'absence de contour et `1` la présence d'un contour. La carte
reste continue. Le choix du seuil qui la transforme en image binaire de limites
revient donc à l'appelant, et le bon seuil dépend du dataset et de l'usage en
aval.

Une prédiction remplit `result.edges`, une charge utile `EdgeMap` contenant un
tableau float32 `(H, W)` dans `[0, 1]` sur le canevas de l'image d'origine.
`.array` renvoie cette carte sous forme NumPy et `.binary(threshold)` renvoie un
masque booléen. `result.boxes` reste vide. Les arguments `conf`, `iou` et
`max_det` n'ont donc aucun effet. `Results.plot()` couvre cette tâche et rend
directement la carte.

## Modèles

Trois familles couvrent la tâche `edge`.

[DexiNed](/docs/models/dexined), le Dense Extreme Inception Network, fusionne
plusieurs sorties latérales dans une carte de probabilités unique et s'exécute
à une résolution native de 352 pixels.

[TEED](/docs/models/teed), le Tiny and Efficient Edge Detector, est un petit
réseau à la même résolution native de 352 pixels. Son pas de sous-échantillonnage
est de 4, contre 16 pour DexiNed. Il accepte donc davantage de valeurs
`imgsz`.

[LibreMODUS](/docs/models/libremodus) produit des contours de type Canny comme
l'une des cibles d'un modèle universel. Il nécessite l'extra `modus` et votre
propre compte Hugging Face authentifié. Il ne propose ni `val()` ni `export()`
et ne participe donc pas aux sections de validation et d'exportation ci-dessous.

## Prédire

LibreYOLO ne publie aucun checkpoint de contours. Les poids DexiNed et TEED
officiellement publiés ont été entraînés sur BIPED, dont les conditions du
dataset limitent l'utilisation à des fins non commerciales. LibreYOLO ne les
reproduit donc pas. Convertissez un checkpoint que votre licence vous autorise
à utiliser, puis chargez le fichier converti par son chemin :

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Le nom du fichier doit comporter le suffixe de tâche `-edge` afin que le
chargeur le reconnaisse. `imgsz` doit être divisible par le pas de
sous-échantillonnage du réseau. Dans le cas contraire, LibreYOLO déclenche une
erreur claire qui nomme le diviseur. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

Pour la validation des contours, chaque image RVB est associée à une carte
monocanale de même nom de base et de même résolution, ainsi qu'à un masque de
validité facultatif.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

La cible est un PNG ou TIF monocanal, et non une visualisation RVB. Les cartes
entières sont divisées par la valeur maximale de leur type. Les cartes
flottantes doivent déjà être finies et comprises dans `[0, 1]`. Les pixels du
masque sont considérés comme valides lorsqu'ils ne sont pas nuls, et ceux
ajoutés pour le remplissage ne contribuent jamais aux métriques.
`edge_invert: true` couvre les sources qui enregistrent des contours noirs sur
fond blanc. Consultez les
[formats de datasets](/docs/reference/dataset-formats) pour le contrat complet.

## Entraîner

Aucune famille de contours de LibreYOLO ne possède d'implémentation
d'entraînement. `train()` déclenche une `NotImplementedError` pour les trois.
Chaque page de modèle indique le script de conversion qui transforme un
checkpoint entraîné ailleurs en un fichier chargeable par LibreYOLO.

## Valider

`val()` rapporte les mesures F de type BSDS. Les prédictions continues sont
d'abord amincies par suppression des non-maxima du gradient dans quatre
directions, puis les pixels de contours prédits et de vérité terrain sont
associés un à un dans une tolérance de distance.

<code-tabs name="val" />

`metrics/ODS` est la mesure F optimale à l'échelle du dataset. Les nombres de
correspondances sont regroupés sur le dataset à chaque seuil, puis la meilleure
de ces mesures F regroupées est rapportée. Elle sert également de `fitness`, la
valeur lue pour sélectionner le meilleur checkpoint. `metrics/OIS` est la
mesure F optimale à l'échelle de l'image, soit la moyenne du meilleur score
propre à chaque image. Chaque image peut donc choisir son seuil.
`metrics/best_threshold` est le seuil unique qui a produit ODS. Il s'agit de la
valeur à réutiliser dans `edges.binary()` pendant l'inférence.

Deux arguments définissent le balayage. `edge_thresholds` est l'ensemble des
seuils essayés, de 0,01 à 0,99 par centièmes par défaut. `edge_max_dist` est la
tolérance de correspondance exprimée comme une fraction de la diagonale de
l'image. Elle vaut `0.0075` par défaut. Une paire plus éloignée ne constitue pas
une correspondance.

## Exporter

Un modèle de contours exporté se recharge par `LibreYOLO()` grâce au suffixe de
son fichier. Un fichier `.onnx` se comporte donc comme un checkpoint et renvoie
le même `Results`.

<code-tabs name="export" />

L'exportation des contours emploie un contrat d'exécution à résolution fixe et à
lot de taille 1. `dynamic` et toute valeur de `batch` différente de 1 sont
refusés, et le graphe exporté émet une seule carte de probabilités fusionnée.
La couverture de chaque format figure sur les pages
[DexiNed](/docs/models/dexined) et [TEED](/docs/models/teed), ainsi que dans la
[matrice d'exportation complète](/docs/reference/export-matrix).
La page [Exporter](/docs/export) présente les arguments acceptés par chaque
format.
