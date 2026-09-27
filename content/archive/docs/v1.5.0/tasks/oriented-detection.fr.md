---
title: Détection orientée
seo_title: Détection orientée dans LibreYOLO
description: >-
  Détectez les objets tournés dans LibreYOLO : les familles qui produisent des
  boîtes orientées, la ligne d'étiquette à quatre coins et les appels de
  prédiction, d'entraînement, de validation et d'export.
lead: >-
  La détection d'objets orientée localise chaque instance avec un rectangle
  tourné plutôt qu'aligné sur les axes, afin d'encadrer précisément un objet
  incliné au lieu d'inclure une grande zone d'arrière-plan. La clé de tâche est
  obb.
keywords:
  - détection boîtes orientées
  - détection objets tournés
  - OBB python
  - dataset DOTA
  - détection objets aérienne
  - IoU orientée
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        # Nécessite l'extra rfdetr : pip install "libreyolo[rfdetr]"

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Le suffixe -obb du nom de fichier sélectionne la tâche, aucun

        # argument task n'est donc nécessaire.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        result = model(SAMPLE_IMAGE, save=True)


        obb = result.obb

        print(obb.xywhr)   # (N, 5) : centre x, centre y, largeur, hauteur,
        radians

        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Coins au lieu des angles
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) points des coins en pixels
        print(obb.xyxyxyxyn.shape)   # les mêmes, normalisés
        print(obb.xyxy.shape)        # (N, 4) boîte alignée englobante
    - label: Un checkpoint plus petit
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Poids DOTA v1.0, 15 classes aériennes à 1024 px. Le graphe orienté
        # est reconnu par les tenseurs du checkpoint, sans argument task.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane, ship, harbor, helicopter et 11 autres
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Continue depuis les poids orientés publiés. data doit pointer vers un

        # dataset dont les lignes d'étiquettes contiennent quatre coins.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Depuis des poids de détection
      language: bash
      code: |
        # Les poids de détection ne prédisent aucun angle, ce transfert est donc
        # explicite. Demander task=obb est ce qui l'autorise.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() renvoie un dictionnaire simple, pas un objet.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX et TorchScript sont ici les cibles validées, en FP32,

        # batch 1, sur un canevas fixe de 1024 par 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory effectue le routage d'après le suffixe du fichier, donc un

        # artefact exporté se charge comme un checkpoint et renvoie le même
        Results.

        model = LibreYOLO("LibreRFDETRs-obb.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Définition

La détection orientée ajoute un nombre à une détection : l'angle. Chaque
instance reçoit un rectangle tourné, une classe et un score. Le gain est la
précision de l'encadrement. Un navire à 45 degrés, le toit d'un entrepôt, une
rangée de camions stationnés : une boîte alignée sur les axes autour de l'un
d'eux contient principalement de l'arrière-plan, et deux boîtes voisines se
chevauchent même lorsque les objets ne le font pas. C'est pourquoi cette tâche
est courante dans l'imagerie aérienne et la mise en page de documents, et
pourquoi son dataset de référence est DOTA.

`obb` est la clé de tâche canonique, et le suffixe `-obb` du nom de fichier d'un
checkpoint la sélectionne. `task=` n'est donc pas nécessaire lors du chargement
de poids publiés.

`predict()` remplit `result.obb`. `.xywhr` est la forme canonique `(N, 5)` :
centre x, centre y, largeur, hauteur et angle en radians qui donne la rotation
du côté de la largeur autour du centre. `.conf` et `.cls` contiennent le score
et l'indice de classe dans `result.names`, et `.id` un identifiant de suivi
lors du tracking. `.xyxyxyxy` convertit chaque ligne en ses quatre coins sous
forme de pixels `(N, 4, 2)`, `.xyxyxyxyn` normalise ces coins et `.xyxy` donne
la boîte alignée sur les axes qui les englobe, à utiliser lorsque le code en
aval ne comprend que les rectangles. `result.boxes` est également rempli avec
la forme alignée sur les axes.

## Modèles

Deux familles prennent cette tâche en charge, et le choix dépend de votre
besoin d'entraînement.

[RF-DETR](/docs/models/rf-detr) est celle qui s'entraîne. Elle prédit, entraîne,
valide et exporte les boîtes orientées, et fournit des checkpoints orientés
publiés dans quatre tailles, n, s, m et l. Elle nécessite son propre extra,
`pip install "libreyolo[rfdetr]"`, et sa page de modèle indique la licence des
poids et leur provenance.

Lisez la section ci-dessous sur ce que ces checkpoints prédisent réellement
avant de bâtir un projet autour d'eux.

[RT-DETRv2](/docs/models/rt-detr) est celle qui dispose de poids aériens. Elle
publie de `LibreRTDETRv2n-obb.pt` à `LibreRTDETRv2x-obb.pt`, les checkpoints
officiels DOTA v1.0 à échelle unique convertis au format de LibreYOLO, qui
couvrent les 15 classes de DOTA à 1024 px. Elle ne nécessite aucun extra en
plus du package de base, le graphe orienté est reconnu à partir des tenseurs du
checkpoint, et la prédiction, la validation ainsi que l'export ONNX et
TorchScript sont tous pris en charge. Ce n'est pas le cas de l'entraînement :
la tâche orientée est réservée à l'inférence dans cette famille, `train()` lève
une erreur et aucun transfert depuis ses poids de détection n'est possible, car
ils utilisent un backbone différent. Le tracking et l'augmentation au moment
du test sont également indisponibles pour les boîtes orientées.

En résumé : pour les catégories DOTA prêtes à l'emploi, choisissez RT-DETRv2.
Pour vos propres étiquettes orientées, choisissez RF-DETR.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

Sachez ce que sont les checkpoints RF-DETR publiés avant de les exécuter.
Malgré le statut de benchmark de référence de DOTA pour cette tâche, ces poids
n'ont pas été entraînés dessus. Les quatre ont été initialisés depuis les poids
de détection RF-DETR et affinés sur un unique dataset Roboflow Universe de
séquences de drone, avec six classes de véhicules : bike, bus, car,
other_vehicle, taxi et truck. Leurs fiches de modèle les décrivent comme des
poids de développement, produits pendant la validation de la prise en charge
de l'entraînement orienté, et précisent qu'ils ne doivent pas être considérés
comme des poids de production ou officiels pour un benchmark.

En pratique, cela signifie qu'ils constituent un point de départ fonctionnel
pour les boîtes orientées sur des véhicules vus du dessus, et pour vérifier le
fonctionnement de bout en bout de votre pipeline. Tout autre domaine exige un
entraînement sur vos propres étiquettes orientées et, pour les catégories
aériennes qui font la réputation de DOTA, les checkpoints RT-DETRv2 sont ceux
qui ont réellement été entraînés sur ces données. `conf` et `max_det`
déterminent la sortie comme pour la détection. Consultez la page sur la
[prédiction](/docs/predict) pour les sources, le streaming et la gestion des
résultats.

## Format du dataset

La disposition est celle de la détection : un fichier d'étiquette `.txt` par
image, trouvé en remplaçant `images` par `labels` dans le chemin de l'image et
en changeant l'extension.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Une ligne contient exactement neuf champs, soit un indice de classe suivi de
quatre coins dans l'ordre :

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Les quatre points sont des nombres flottants normalisés dans `[0, 1]` et
doivent former un rectangle orienté non dégénéré. Aucun angle n'est stocké dans
le fichier d'étiquette : le chargeur dérive la forme canonique `xywhr` des
coins. Le parseur est strict par défaut et rejette les coordonnées hors limites,
tandis que l'ingestion pour le dataset et la validation peut d'abord les borner
à `[0, 1]` pour des étiquettes par ailleurs valides situées à la limite d'un
recadrage, avant de toujours rejeter les boîtes dégénérées.

L'analyse des lignes dépend de la tâche. Neuf champs désignent une boîte
orientée uniquement en mode `obb` ; en mode `segment`, la même ligne est lue
comme un polygone à quatre points.

Le YAML est celui de la détection :

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

Le JSON COCO natif se charge également, avec une correspondance `annotations`
entre le nom du split et le fichier JSON. Les annotations sont lues dans
l'ordre de priorité suivant : un champ `obb` de huit coins exprimés en pixels,
un champ `obb` de forme `[cx, cy, w, h, angle]` avec l'angle en radians, un
polygone `segmentation` ou un RLE réajusté à son rectangle d'aire minimale, ou
un simple `bbox` COCO, traité comme un rectangle aligné sur les axes et
canonisé en `xywhr`.

Le parseur de ligne canonique est
`libreyolo.data.parse_yolo_obb_label_line`.

## Entraîner

<code-tabs name="train" />

Pour cette tâche, l'entraînement utilise RF-DETR. Il continue par défaut depuis
un checkpoint `-obb` publié. Démarrer depuis des poids de détection constitue
un transfert volontaire : ces poids ne prédisent aucun angle, et le passage de
`task=obb` autorise ce remplacement. Maintenez `lr0` à `1e-4` ou en dessous,
comme pour les autres tâches de la famille. Les checkpoints orientés de
RT-DETRv2 ne peuvent pas faire l'objet d'un fine-tuning ; utilisez-les tels
quels ou entraînez un modèle RF-DETR sur vos propres étiquettes. Consultez la
page sur l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un simple dictionnaire de clés `metrics/`. L'association
utilise l'IoU orientée, calculée entre les rectangles orientés plutôt qu'entre
leurs boîtes englobantes alignées sur les axes. Une prédiction correctement
placée mais au mauvais angle est donc comptée comme un échec.

<code-tabs name="val" />

`metrics/mAP50-95` est la précision moyenne calculée sur les seuils d'IoU de
0.50 à 0.95 par pas de 0.05, et constitue la mesure principale. Contrairement
au chemin COCO utilisé par la détection, cette tâche respecte `iou_thresholds`
dans la configuration de validation, ce qui permet de modifier le balayage.
`metrics/mAP50` et `metrics/mAP75` sont les versions à seuil unique.
`metrics/precision` et `metrics/recall` sont la précision et le rappel réels à
une IoU de 0.50, lus au point de fonctionnement le plus permissif : chaque
prédiction ayant dépassé le seuil de confiance est comptée, et ce seuil vaut
0.001 par défaut pendant la validation. Augmenter `conf` les déplace donc,
tandis que les valeurs mAP, qui utilisent l'ensemble de la courbe
précision-rappel, restent stables. Quatre de ces mesures se répètent avec le
suffixe `(OBB)`, `metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`,
`metrics/precision(OBB)` et `metrics/recall(OBB)`, ce qui permet à l'appelant de
distinguer un résultat orienté d'un résultat aligné sur les axes lorsqu'ils
figurent dans la même table. `metrics/mAP75` n'a aucun équivalent suffixé.

Deux options n'ont aucun effet sur cette tâche. `save_json` et `save_plots`
sont acceptées et journalisent un avertissement : les sorties de prédictions
orientées et les graphiques de validation ne sont pas implémentés.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. La couverture des formats varie selon
la tâche au sein d'une même famille, et la matrice de la page du modèle est
générée depuis l'ensemble validé et indique pourquoi une cible est
indisponible. Consultez la page sur l'[export et le déploiement](/docs/export)
pour les formats, leurs extras et leurs contraintes.
