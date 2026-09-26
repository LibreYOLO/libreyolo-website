---
title: Détection d'objets
seo_title: Détection d'objets dans LibreYOLO
description: >-
  Détecter des objets sous forme de boîtes alignées sur les axes dans LibreYOLO
  : familles qui couvrent la tâche, format des annotations et appels de
  prédiction, d'entraînement, de validation et d'exportation.
lead: >-
  La détection d'objets localise chaque instance d'objet dans une image et
  renvoie pour chacune un rectangle aligné sur les axes, une étiquette de classe
  et un score. La clé de tâche est detect.
keywords:
  - détection objets python
  - détecter objets image
  - détection boîtes englobantes
  - bibliothèque détection objets MIT
  - alternative YOLO
  - entraîner détecteur objets
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Une autre famille, le même appel'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique s'oriente selon le checkpoint et chaque détecteur renvoie

        # le même objet Results. Changer de famille ne demande donc qu'une
        ligne.

        model = LibreYOLO("LibreDFINEn.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy.shape)
    - label: Vidéos et flux
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # indice de webcam, flux RTSP ou liste .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco128.yaml télécharge un échantillon de 128 images à la première

        # utilisation. Pour une véritable exécution, utilisez le YAML de votre
        dataset.

        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Définition

La détection d'objets indique où se trouve chaque objet et ce qu'il est. Une
image entre, une ligne sort par instance : quatre nombres pour le rectangle, un
indice de classe et un score. Elle ne décrit ni la forme des pixels, ni
l'orientation, ni les parties, ce qui la distingue de la
[segmentation d'instances](/docs/tasks/instance-segmentation), des
[boîtes orientées](/docs/tasks/oriented-detection) et de la
[pose](/docs/tasks/pose-estimation).

`detect` est la clé de tâche canonique et la valeur par défaut. Un checkpoint
dont le nom de fichier ne comporte aucun suffixe de tâche se charge comme un
détecteur.

`predict()` remplit `result.boxes`. `.xyxy` donne les coins en pixels sur le
canevas de l'image d'origine, `.conf` le score et `.cls` l'indice de classe
dans `result.names`. `.xywh`, `.xyxyn` et `.xywhn` sont des vues dérivées des
mêmes lignes, et `.id` contient un identifiant de suivi lorsqu'un tracker est
attaché. Parcourir un objet `Boxes` produit des tranches d'une ligne.
`box.cls`, `box.conf` et `box.xyxy` fonctionnent donc pour chaque détection.

## Modèles

Douze familles peuvent être entraînées et effectuer des prédictions :
[YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[D-FINE](/docs/models/d-fine), [DEIM](/docs/models/deim),
[Dome-DETR](/docs/models/dome-detr), [YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) et [PicoDet](/docs/models/picodet). YOLOv9 et
RF-DETR sont les deux familles vedettes, qui reçoivent les fonctionnalités en
premier. RF-DETR nécessite son propre extra,
`pip install "libreyolo[rfdetr]"`. Les autres s'exécutent avec le paquet de
base.

Onze autres familles effectuent la prédiction, la validation et l'exportation,
mais leur méthode `train()` déclenche une `NotImplementedError` :
[LW-DETR](/docs/models/lw-detr), [DETR](/docs/models/detr),
[Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) et
[EfficientDet](/docs/models/efficientdet).

La lignée Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) et
[YOLOv4](/docs/models/yolov4), est conservée comme une pièce figée. La
prédiction, la validation et l'exportation fonctionnent, mais pas
l'entraînement.

Un groupe distinct reçoit sa liste de classes à l'exécution plutôt que depuis le
checkpoint. Il détecte donc des noms jamais vus pendant l'entraînement :
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) et [OV-DEIM](/docs/models/ov-deim),
ainsi que les familles vision-langage [Florence-2](/docs/models/florence-2),
[Kosmos-2](/docs/models/kosmos-2), [Qwen3-VL](/docs/models/qwen3-vl),
[SmolVLM2](/docs/models/smolvlm2), [InternVL3](/docs/models/internvl3),
[LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) et
[LibreMODUS](/docs/models/libremodus). Ces modèles se chargent par leur propre
fabrique et leurs propres extras. Chaque page de modèle présente l'appel exact.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

`conf` définit le seuil de confiance et `max_det` limite le nombre de lignes.
`iou` est le seuil de NMS. Il n'a donc d'effet que sur une famille qui applique
la NMS. RF-DETR et la tête YOLOv9 de bout en bout décodent un ensemble fixe de
prédictions et l'ignorent. Consultez la page [prédiction](/docs/predict) pour
les sources, le streaming et la gestion des résultats.

## Format du dataset

Chaque image possède un fichier d'étiquettes `.txt`, trouvé en remplaçant
`images` par `labels` dans le chemin de l'image et en changeant l'extension.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Chaque ligne contient exactement cinq champs : un indice de classe suivi d'une
boîte normalisée sous forme de centre et de taille.

```text
<class_id> <cx> <cy> <w> <h>
```

Les coordonnées sont des nombres flottants dans `[0, 1]`, relatifs à la largeur
et à la hauteur de l'image d'origine. `w` et `h` doivent être positifs. Un
fichier d'étiquettes absent ou vide signifie que l'image ne contient aucun
objet. Les lignes ne comportent ni confiance ni identifiant de suivi.

Le fichier YAML nomme les partitions et les classes :

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` et `val` peuvent être des répertoires d'images, des fichiers `.txt` qui
énumèrent des images ou des listes de ces deux formes. `nc` est facultatif et
doit correspondre à `names` lorsqu'il est présent. Le JSON COCO natif fonctionne
également. Ajoutez une association `annotations` entre le nom de partition et
le fichier JSON. Le chemin de la partition indique alors la racine des images.
Lorsque `names` est présent, il définit les identifiants des étiquettes. Les
noms de catégories du JSON doivent donc lui correspondre.

## Entraîner

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` et `lr0` sont les premiers arguments à ajuster.
`lr0` est celui qui ne se transfère pas d'une famille à l'autre. Un taux toléré
par un détecteur convolutif fera diverger un Transformer. Reprenez donc la
valeur de la page du modèle plutôt que celle de l'exemple d'une autre famille.
Une famille peut aussi ignorer complètement un argument, ce que sa page
indique. Consultez la page [entraînement](/docs/train) pour les datasets, les
augmentations, le multi-GPU et les systèmes de journalisation.

## Valider

`val()` renvoie un dictionnaire ordinaire de clés `metrics/`, calculées avec
l'évaluation COCO sur la partition `val` du fichier YAML du dataset.

<code-tabs name="val" />

`metrics/mAP50-95` est la précision moyenne, moyennée sur les seuils d'IoU de
0,50 à 0,95. C'est le nombre principal. `metrics/mAP50` et `metrics/mAP75`
sont les versions à seuil unique. `metrics/mAP_small`,
`metrics/mAP_medium` et `metrics/mAP_large` répartissent la même moyenne selon
l'aire de l'objet. `metrics/AR1`, `metrics/AR10`, `metrics/AR100`,
`metrics/AR_small`, `metrics/AR_medium` et `metrics/AR_large` sont les valeurs
correspondantes de rappel moyen. `metrics/AR_max_det` et `metrics/max_det`
enregistrent la limite de détections utilisée pendant l'exécution.

Lisez attentivement `metrics/precision` et `metrics/recall` pour cette tâche.
Elles sont conservées pour la rétrocompatibilité et constituent des alias, pas
un point de fonctionnement. `metrics/precision` contient la même valeur que
`metrics/mAP50-95`, et `metrics/recall` la même valeur que `metrics/AR100`.
Les tracer comme une paire précision-rappel rapporte donc deux fois le même
nombre. Quatre clés sont aussi répétées avec le suffixe `(B)`, pour boîte, afin
qu'une clé de détection se lise de la même manière sur un modèle qui prédit
aussi des masques : `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
`metrics/precision(B)` et `metrics/recall(B)`.

## Exporter

<code-tabs name="export" />

Un artefact exporté se recharge par `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie le même `Results`. La couverture des formats varie selon la famille.
La matrice de chaque page de modèle est générée depuis l'ensemble validé plutôt
que saisie manuellement. Consultez la page
[exporter et déployer](/docs/export) pour les formats, leurs extras et leurs
contraintes.
