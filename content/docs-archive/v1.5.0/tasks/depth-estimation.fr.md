---
title: Estimation de la profondeur
seo_title: Estimation monoculaire de la profondeur dans LibreYOLO
description: >-
  Prédire une carte dense de profondeur relative depuis une image dans
  LibreYOLO. Comparer les familles de profondeur, lire les métriques et exporter
  un modèle.
lead: >-
  L'estimation de la profondeur prédit la distance de chaque pixel à la caméra à
  partir d'une seule image. LibreYOLO l'expose comme la tâche depth, qui renvoie
  une carte dense de profondeur inverse relative sur le canevas de l'image
  d'origine.
keywords:
  - estimation profondeur monoculaire python
  - carte profondeur image unique
  - modèle profondeur relative
  - depth anything libreyolo
  - prédiction dense profondeur
last_verified: 1.5.0
snippets:
  predict:
    - label: Prédire une carte de profondeur
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) sur le canevas d'origine
        print(depth.min, depth.max, depth.mean)
    - label: Manipuler les valeurs
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # une valeur élevée est proche ; aucune unité
        ni échelle métrique

        gray = depth.normalized() # remis à l'échelle dans [0, 1] pour la
        visualisation

        print(raw.shape, float(gray.max()))
    - label: Une solution compacte
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Même contrat de tâche, avec un réseau bien plus petit conçu pour la
        périphérie.

        model = LibreYOLO("LibreZipDepthb-depth.pt")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Définition

La tâche `depth` prédit une valeur par pixel depuis une image RVB unique.
LibreYOLO définit cette valeur comme une profondeur inverse relative. Une valeur
élevée signifie que le point est proche de la caméra, et les nombres ne
possèdent aucune unité métrique ni aucune échelle commune à deux images.
Comparer les profondeurs de deux pixels d'une même prédiction a un sens. La
comparaison avec une valeur provenant d'une autre image n'en a pas.

Une prédiction remplit `result.depth_map`, une charge utile `DepthMap` contenant
un tableau `(H, W)` sur le canevas de l'image d'origine. `.min`, `.max` et
`.mean` lisent les valeurs finies, tandis que `.normalized()` remet la carte à
l'échelle dans `[0, 1]` pour l'affichage. `result.boxes` reste vide. Les
arguments `conf`, `iou` et `max_det` n'ont donc aucun effet, et `save=True`
écrit une image de la carte avec une palette de couleurs plutôt qu'une photo
annotée.

## Modèles

Six familles couvrent la tâche `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) associe un encodeur DINOv2
à un décodeur DPT et constitue ici le choix polyvalent par défaut. La licence
compte autant que la précision dans le choix de la taille. Le checkpoint Small
est sous licence Apache-2.0, tandis que Base et Large sont réservés à un usage
non commercial. Consultez le tableau des checkpoints de sa page avant de
choisir.

[Depth Anything 3](/docs/models/depth-anything-3) porte le checkpoint
DA3MONO-LARGE, un Transformer ordinaire sans spécialisation architecturale pour
la profondeur.

[ZipDepth](/docs/models/zipdepth) constitue le niveau compact. Il s'agit d'un
CNN reparamétrable distillé depuis Depth Anything V2 Large, avec un second
checkpoint dont le décodeur évite les opérations gather et unfold pour les
compilateurs NPU qui ne les proposent pas.

[MiDaS](/docs/models/midas) est la lignée de travaux qui a établi le protocole
zero-shot de profondeur relative servant à mesurer les autres familles. C'est
la seule famille de profondeur que LibreYOLO ne republie pas. La demande d'un
checkpoint télécharge la ressource officielle depuis la version GitHub des
auteurs et contrôle une empreinte SHA-256 fixée.

[LibreMODUS](/docs/models/libremodus) atteint la profondeur comme l'une des
cibles d'un modèle universel, plutôt que par une tête dédiée. Il nécessite
l'extra `modus` et votre propre compte Hugging Face authentifié, et ne propose
ni `val()` ni `export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) génère la carte de profondeur
comme une image par un décodage de diffusion, depuis le même checkpoint 7B qui
couvre ses six autres tâches. Il nécessite l'extra `sensenova`, et ses poids
sont réservés à un usage non commercial. Leur licence figure sur sa page.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement, à l'exception des deux familles signalées plus haut.

<code-tabs name="predict" />

La résolution d'entrée est contrainte par famille. Depth Anything V2 et
Depth Anything 3 reposent sur une grille de patches DINOv2. `imgsz` doit donc
être divisible par 14, ce que LibreYOLO vérifie avant l'exécution.
`Results.plot()` ne couvre pas cette tâche. Cette méthode est définie uniquement
pour les normales de surface et les contours. Consultez la page
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

Pour la validation de la profondeur, chaque image est associée à une carte de
profondeur dense monocanale de même résolution, trouvée en remplaçant le
répertoire de profondeur dans le chemin de l'image.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Les cartes sont des PNG ou TIF monocanaux, ou des fichiers `.npy`. Les valeurs
représentent directement la profondeur dans une unité que le dataset conserve
de façon cohérente. Les pixels nuls, négatifs, NaN et infinis signalent des
échantillons non valides exclus des métriques. Les cartes entières sont
divisées par `depth_scale`, qui vaut `256.0` par défaut conformément à la
convention des PNG 16 bits. Les cartes flottantes `.npy` sont utilisées telles
quelles. `depth_stem_suffix` et `depth_mask_suffix` couvrent les datasets qui
nomment différemment leurs fichiers de profondeur ou leurs masques de validité.
Consultez les [formats de datasets](/docs/reference/dataset-formats) pour le
contrat complet.

## Entraîner

Aucune famille de profondeur de LibreYOLO ne possède d'implémentation
d'entraînement. `train()` déclenche une `NotImplementedError` pour les six.
Chaque page de modèle indique le script de conversion qui transforme un
checkpoint entraîné dans le projet amont en un fichier chargeable par
LibreYOLO.

## Valider

`val()` exécute le validateur de profondeur partagé. La profondeur relative
n'a pas d'échelle absolue. Chaque prédiction est donc d'abord ajustée à
l'inverse de sa vérité terrain avec une échelle et un décalage calculés par
moindres carrés pour chaque image, puis reconvertie en profondeur. Toutes les
métriques ci-dessous sont calculées par image sur cette carte alignée et
moyennées sur le dataset, en ne comptant que les pixels que le dataset marque
comme valides.

<code-tabs name="val" />

`metrics/abs_rel` est l'erreur relative absolue moyenne, soit le résidu divisé
par la profondeur de vérité terrain. Une valeur faible est préférable.
`metrics/rmse` est la racine de l'erreur quadratique moyenne dans l'unité de
profondeur propre au dataset. Là encore, une valeur faible est préférable.
`metrics/delta1`, `metrics/delta2` et `metrics/delta3` sont les précisions par
seuil : la fraction de pixels valides dont le rapport à la vérité terrain, pris
dans le sens qui donne la valeur la plus élevée, reste inférieur à 1,25,
1,25 au carré et 1,25 au cube. Une valeur élevée est donc préférable.
`metrics/delta1` sert également de `fitness`, la valeur lue pour sélectionner le
meilleur checkpoint.

## Exporter

Un modèle de profondeur exporté se recharge par `LibreYOLO()` grâce au suffixe
de son fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un
checkpoint et renvoie le même `Results`, avec `depth_map` à la place des boîtes.

<code-tabs name="export" />

La couverture diffère selon la famille. Depth Anything 3 refuse tout format
extérieur à son ensemble validé au lieu de tenter une conversion non validée.
Consultez la page du modèle et la
[matrice d'exportation complète](/docs/reference/export-matrix) avant de choisir
une cible. LibreMODUS et SenseNova-Vision ne s'exportent pas.
La page [Exporter](/docs/export) répertorie les arguments acceptés par chaque
format.
