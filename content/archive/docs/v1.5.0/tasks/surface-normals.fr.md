---
title: Normales de surface
seo_title: Estimation des normales de surface dans LibreYOLO
description: >-
  Prédire un champ dense de normales de surface depuis une image dans LibreYOLO.
  Comprendre la convention du repère caméra, valider l'erreur angulaire et
  exporter un modèle.
lead: >-
  L'estimation des normales de surface prédit la direction vers laquelle fait
  face chaque surface visible. LibreYOLO l'expose comme la tâche normal, qui
  renvoie un champ dense de vecteurs unitaires sur le canevas de l'image
  d'origine.
keywords:
  - estimation normales surface python
  - carte normales depuis image
  - géométrie monoculaire
  - métrique erreur angulaire
  - prédiction dense normales
last_verified: 1.5.0
snippets:
  predict:
    - label: Prédire un champ de normales
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # vecteurs unitaires float32 (H, W, 3)

        normals.assert_normalized()    # déclenche une erreur si un pixel n'est
        pas unitaire
    - label: Lire un pixel
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # Repère caméra OpenCV : +x à droite, +y vers le bas, +z dans la scène.
        # Une surface face à la caméra est proche de (0, 0, -1).
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: Enregistrer la visualisation
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # plot() rend le champ ; cette méthode est définie pour les normales et
        les contours.

        result.plot().save("normals.png")
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # degrés
        print(metrics["metrics/median_angular_error"])   # degrés
        print(metrics["metrics/within_11_25"])           # pourcentage de pixels
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Définition

La tâche `normal` prédit depuis une image RVB un vecteur unitaire à trois
composantes par pixel, qui indique la direction vers laquelle fait face la
surface de ce pixel. Contrairement à la profondeur, la sortie n'a pas d'échelle
libre. Deux prédictions sont donc directement comparables sans alignement.

Une prédiction remplit `result.normal_map`, une charge utile `NormalMap` qui
contient un tableau float32 `(H, W, 3)` sur le canevas de l'image d'origine,
également accessible sous `result.normals`. Les vecteurs emploient le repère
caméra OpenCV de LibreYOLO, avec `+x` vers la droite, `+y` vers le bas et `+z`
dans la scène. Ils font face à la caméra. Une surface fronto-parallèle vaut donc
`(0, 0, -1)`. `.assert_normalized()` vérifie que chaque pixel est fini et de
longueur unitaire dans une certaine tolérance. `result.boxes` reste vide. Les
arguments `conf`, `iou` et `max_det` n'ont donc aucun effet, et
`Results.plot()` couvre cette tâche.

## Modèles

Deux familles couvrent la tâche `normal`.

[MoGe-2](/docs/models/moge-2) est la famille dédiée. Ce modèle de géométrie
monoculaire à propagation unique existe en trois tailles d'encodeur. LibreYOLO
ne copie pas ces checkpoints dans sa propre organisation. Le chargement
télécharge la taille correspondante depuis les dépôts officiels à une révision
fixée, puis la vérifie avec une empreinte SHA-256 enregistrée.

[LibreMODUS](/docs/models/libremodus) produit des normales comme l'une des
cibles d'un modèle universel et peut recevoir une carte de profondeur plutôt
qu'une image RVB en entrée. Il nécessite l'extra `modus` et votre propre compte
Hugging Face authentifié. Il ne propose ni `val()` ni `export()`, et ne
participe donc pas aux sections de validation et d'exportation ci-dessous.

## Prédire

Les poids MoGe-2 sont téléchargés à la première utilisation et mis en cache
localement.

<code-tabs name="predict" />

`imgsz` doit être divisible par la taille de patch de l'encodeur ViT, ce que
LibreYOLO vérifie avant le début de l'exécution. La prédiction sur une liste
d'images effectue une propagation par image. Cette tâche ne possède aucun
parcours rapide par lots empilés. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

Pour la validation des normales, chaque image est associée à un PNG 16 bits à
trois canaux, de même nom de base et de même résolution, ainsi qu'à un masque de
validité facultatif.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

Le PNG cible doit comporter exactement trois canaux `uint16` enregistrés en
RVB. Le décodage suit `n = png / 65535 * 2 - 1`, puis renormalise chaque
vecteur. Les vecteurs décodés emploient le même repère caméra OpenCV que les
prédictions. Un pixel de masque est valide s'il n'est pas nul. Sans fichier de
masque, tout vecteur décodé fini et non nul est valide. Les pixels cibles non
valides et ceux ajoutés pour le remplissage sont conservés en interne sous la
forme `(0, 0, 0)` et ne contribuent jamais aux métriques. Consultez les
[formats de datasets](/docs/reference/dataset-formats) pour le contrat complet.

## Entraîner

Aucune des deux familles de normales ne possède d'implémentation
d'entraînement. `train()` déclenche une `NotImplementedError` dans les deux
cas. La page de MoGe-2 renvoie vers ses checkpoints officiels fixés pour la
prédiction, la validation et l'exportation.

## Valider

`val()` mesure l'angle entre chaque vecteur prédit et son vecteur de vérité
terrain, sur les pixels que le dataset marque comme valides.

<code-tabs name="val" />

`metrics/mean_angular_error` et `metrics/median_angular_error` représentent cet
angle en degrés. Les valeurs faibles sont préférables.
`metrics/within_11_25`, `metrics/within_22_5` et `metrics/within_30` sont les
pourcentages de pixels valides dont l'erreur angulaire se situe respectivement
dans les limites de 11,25, 22,5 et 30 degrés. Les valeurs élevées sont donc
préférables. Notez l'unité : ces trois valeurs sont des pourcentages et non des
fractions. `fitness` correspond à `metrics/within_11_25` divisé par 100, ce qui
place la sélection du meilleur checkpoint sur la même échelle `[0, 1]` que
toutes les autres tâches.

## Exporter

Un modèle de normales exporté se recharge par `LibreYOLO()` grâce au suffixe de
son fichier. Un fichier `.onnx` se comporte donc comme un checkpoint et renvoie
le même `Results`.

<code-tabs name="export" />

L'exportation des normales emploie un contrat d'exécution à résolution fixe et
à lot de taille 1. `dynamic` et toute valeur de `batch` différente de 1 sont
refusés, et `imgsz` doit être divisible par la taille de patch de l'encodeur.
La couverture de chaque format figure sur la
[page de MoGe-2](/docs/models/moge-2) et dans la
[matrice d'exportation complète](/docs/reference/export-matrix).
La page [Exporter](/docs/export) présente les arguments acceptés par chaque
format.
