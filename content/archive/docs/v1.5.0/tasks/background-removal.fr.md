---
title: Suppression de l'arrière-plan
seo_title: Suppression de l'arrière-plan dans LibreYOLO
description: >-
  Détourer un sujet de son arrière-plan dans LibreYOLO. Prédire un cache alpha
  progressif, écrire un PNG transparent et valider avec la MAE et la mesure S.
lead: >-
  La suppression de l'arrière-plan sépare un sujet de tout ce qui se trouve
  derrière lui. LibreYOLO l'expose comme la tâche matte, qui renvoie une valeur
  alpha progressive par pixel plutôt qu'un masque binaire de premier plan.
keywords:
  - suppression arrière-plan python
  - modèle alpha matting
  - segmentation image dichotomique
  - détourage png transparent
  - cache alpha progressif
last_verified: 1.5.0
snippets:
  predict:
    - label: Prédire un cache
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        matte = result.matte

        print(matte.array.shape, matte.array.dtype)   # float32 (H, W) dans [0,
        1]
    - label: Écrire un PNG transparent
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() compose la source avec le cache comme canal alpha.
        result.save("subject.png")

        rgba = result.cutout()   # le même tableau uint8 (H, W, 4) en mémoire
        print(rgba.shape)
    - label: Composer sur un nouvel arrière-plan
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # blanc

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreBiRefNetl-matte.pt")


        # Un répertoire contenant images/ et un répertoire de caches peut

        # remplacer un fichier YAML de dataset.

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])        # une valeur faible est préférable

        print(metrics["metrics/Smeasure"])   # fitness, une valeur élevée est
        préférable
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Définition

La tâche `matte` prédit une valeur alpha par pixel depuis une image RVB : `1`
représente entièrement le premier plan et `0` entièrement l'arrière-plan. La
valeur est continue plutôt que binaire, ce qui constitue précisément l'intérêt
de la tâche. Un seuil à 0,5 suffit à produire un masque binaire. Le cache
progressif conserve en plus la couverture partielle des cheveux, de la fourrure
et des contours floutés par le mouvement, qu'un masque binaire perd.

Une prédiction remplit `result.matte`, une charge utile `Matte` contenant un
tableau float32 `(H, W)` dans `[0, 1]` sur le canevas de l'image d'origine,
accessible sous forme NumPy avec `.array`. `result.cutout()` compose l'image
source avec cet alpha dans un tableau RGBA uint8 `(H, W, 4)`, et
`result.save(path)` écrit le même contenu dans un PNG à arrière-plan
transparent. `result.boxes` reste vide. Les arguments `conf`, `iou` et
`max_det` n'ont donc aucun effet.

## Modèles

Deux familles couvrent `matte` et partagent le même parcours de propagation.

[BiRefNet](/docs/models/birefnet) est le réseau à référence bilatérale autour
duquel la tâche est construite. Il est publié ici sous la forme d'un checkpoint
de niveau Swin-L.

[FeyNobg](/docs/models/feynobg) est la variante approfondie de Feyn Inc. :
l'architecture de BiRefNet dont le troisième étage Swin passe de 18 à 24 blocs,
puis réentraînée. LibreYOLO réutilise le parcours de propagation, le
prétraitement et la sortie à un logit de BiRefNet. La prédiction, la validation
et la gestion des checkpoints se comportent donc de façon identique. Les poids
et l'identité de famille restent propres à FeyNobg.

Les deux familles possèdent des licences de poids différentes. Elles sont
indiquées sur les pages des modèles. La licence du dépôt Hugging Face du
checkpoint précis fait autorité.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

Les deux familles s'exécutent sur un canevas natif fixe de 1 024 x 1 024, puis
redimensionnent le cache vers l'image d'origine. Aucune autre résolution n'est
prise en charge, car les tables de position relative du backbone Swin sont
liées à cette taille. Une incompatibilité les interpole mal au lieu de
déclencher une erreur. `Results.save()` n'est défini que pour les résultats de
cache et nécessite l'image source, qu'il recharge depuis `Results.path` sauf si
vous en fournissez une. Consultez la page [prédiction](/docs/predict) pour les
sources, le streaming et la gestion des résultats.

## Format du dataset

Pour la validation, chaque image RVB est associée à un cache alpha de vérité
terrain monocanal portant le même nom de base, où 0 représente l'arrière-plan
et 255 le premier plan.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Il suffit de transmettre cette racine dans `data=`. Le répertoire des caches est
détecté automatiquement parmi `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` et
`alpha/`. Un fichier YAML de dataset constitue l'autre possibilité. Il contient
`path` ainsi que `val_images` et `val_mattes`, qui désignent des répertoires
relatifs :

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` et `names` sont des valeurs réservées du schéma. Un modèle de cache renvoie
`Results.matte`, pas des détections. Les valeurs du cache sont lues comme un
alpha dans `[0, 1]` par division par 255. Un cache dont la forme diffère du
canevas de prédiction est redimensionné par interpolation bilinéaire. Consultez
les [formats de datasets](/docs/reference/dataset-formats) pour le contrat
complet.

## Entraîner

Aucune des deux familles de cache ne possède d'implémentation d'entraînement.
`train()` déclenche une `NotImplementedError` pour chacune, et la prise en
charge couvre seulement la prédiction, la validation et l'exportation. Chaque
page de modèle nomme le projet amont qui fournit le code d'entraînement et le
script de conversion qui réimporte un checkpoint.

## Valider

`val()` exécute la propre méthode `predict` du modèle. La validation emploie
donc exactement le prétraitement de la famille, et les deux métriques sont
calculées sur le canevas de l'image d'origine.

<code-tabs name="val" />

`metrics/MAE` est l'erreur absolue moyenne par rapport à l'alpha de vérité
terrain, dans `[0, 1]`. Une valeur faible est préférable. `metrics/Smeasure` est
la mesure S de Fan et al. (ICCV 2017), une similarité structurelle qui récompense
la forme correcte du sujet et de ses trous, ce qu'une moyenne par pixel ne
reflète pas. Une valeur élevée est préférable. La mesure S sert également de
`fitness`, la valeur lue pour sélectionner le meilleur checkpoint. Aucune de
ces métriques ne dépend de la résolution.

## Exporter

Un modèle de cache exporté se recharge par `LibreYOLO()` grâce au suffixe de
son fichier. L'artefact se comporte donc comme un checkpoint et renvoie le même
`Results`.

<code-tabs name="export" />

TorchScript est le parcours validé pour cette tâche. La conversion ONNX
s'exécute mais n'a pas franchi le même seuil de parité. Les autres formats ne
sont pas disponibles. La couverture de chaque format figure sur les pages
[BiRefNet](/docs/models/birefnet) et [FeyNobg](/docs/models/feynobg), ainsi que
dans la [matrice d'exportation complète](/docs/reference/export-matrix).
