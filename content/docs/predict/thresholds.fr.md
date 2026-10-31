---
title: Seuils et filtrage
seo_title: 'conf, iou et max_det dans LibreYOLO'
description: >-
  Ce que font conf, iou, max_det, classes et agnostic_nms lors de la prédiction,
  et les familles qui ignorent iou faute de NMS propre.
lead: "Quatre arguments déterminent les prédictions conservées\_: conf, iou, max_det et classes. Seuls deux s'appliquent à chaque famille, car un prédicteur d'ensembles décode un ensemble fixe de requêtes et n'exécute jamais de NMS."
keywords:
  - seuil conf yolo
  - seuil iou nms
  - max_det
  - filtrer classes détection python
  - agnostic nms
  - detr sans nms
  - seuil confiance détection
  - filtrage classes inférence
last_verified: 1.6.0
verification: >-
  Valeurs par défaut citées depuis InferenceRunner.__call__ dans
  libreyolo/models/base/inference.py. Comportement NMS par famille lu dans
  chaque module de libreyolo/postprocess/ et recoupé avec _is_nms_free_family
  dans libreyolo/backends/base.py. Filtrage des classes lu dans
  InferenceRunner._apply_classes_filter et _wrap_results. agnostic_nms lu dans
  check_agnostic_nms et postprocess_max_det dans libreyolo/utils/predict_args.py
  et dans l'étape agnostique de InferenceRunner._wrap_results. Gestion du
  vocabulaire ouvert lue dans NMS_THRESHOLD dans
  libreyolo/models/openvocab/base.py. Valeurs par défaut de validation lues dans
  BaseModel.val.
snippets:
  basic:
    - label: Les quatre arguments
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # conserver les prédictions dont le score atteint au moins cette valeur
            iou=0.45,       # seuil de chevauchement NMS, lorsqu'une NMS s'exécute
            max_det=300,    # limite par image
            classes=None,   # ou une liste d'identifiants de classe
        )
        print(len(result.boxes))
    - label: Balayer conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Filtrer des classes précises
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # Les identifiants de classe indexent model.names. Sur COCO, 0 désigne
        person.

        result = model(SAMPLE_IMAGE, classes=[0])


        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Trouver l'identifiant d'un nom
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou sur une famille sans NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR décode un ensemble fixe de requêtes, iou ne change donc rien
        ici.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Même nombre dans les deux cas. conf et max_det sont les réglages
        efficaces.

        print(len(loose.boxes), len(tight.boxes))
  agnostic:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        per_class = model(SAMPLE_IMAGE)

        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)


        # Les boîtes qui se chevauchent et de classes différentes se réduisent à
        celle au score le plus élevé.

        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
---

## Les quatre arguments

| Argument | Valeur par défaut | S'applique à |
|---|---|---|
| `conf` | `0.25` | Chaque famille |
| `iou` | `0.45` | Familles qui exécutent une suppression non maximale |
| `max_det` | `300` | Chaque famille |
| `classes` | `None` | Chaque famille |

<code-tabs name="basic" />

Deux de ces arguments sont universels et deux ne le sont pas. C'est
l'information essentielle à connaître avant de régler quoi que ce soit.

La validation utilise délibérément d'autres valeurs par défaut : `val()`
s'exécute avec `conf=0.001` et `iou=0.6`, car la précision moyenne est calculée
sur une courbe précision-rappel complète et un seuil de 0.25 la tronquerait.

## conf

`conf` est le score en dessous duquel une prédiction est supprimée. Il
s'applique à chaque famille, y compris à celles qui n'exécutent jamais de NMS,
et constitue le premier réglage à modifier lorsque les détections sont trop
nombreuses ou trop rares.

La valeur par défaut `0.25` convient à l'examen d'images. L'alimentation d'un
système downstream demande généralement une valeur supérieure, tandis qu'une
mesure d'exactitude exige une valeur bien inférieure.

La validation de détection expose `metrics/best_conf` et des seuils par classe fondés sur le F1 à IoU 0.50. Utilisez-les comme points de départ mesurés pour les seuils de déploiement ; consultez la [validation](/docs/train/validation). La segmentation n'expose pas ces clés.

## iou

`iou` est le chevauchement au-delà duquel la suppression non maximale retire
la bounding box au score le plus faible parmi deux bounding boxes de même
classe. Il n'a de sens que si la famille exécute effectivement une suppression.

Un prédicteur d'ensembles décode un nombre fixe de requêtes et conserve celles
dont le score est le plus élevé. Les doublons sont supprimés dans l'architecture
pendant l'entraînement et non lors d'une étape de post-traitement, aucun seuil
ne peut donc être réglé. Les familles suivantes acceptent `iou` pour assurer
la parité de l'API, mais l'ignorent :

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR et la tête de bout en bout
de YOLOv9. Les variantes construites sur ces décodeurs héritent de ce
comportement.

<code-tabs name="nmsfree" />

La plupart l'indiquent dans les docstrings de leur post-traitement, mais aucun
avertissement n'est levé à l'exécution. Un balayage de `iou` sur RF-DETR produit
donc une ligne plate plutôt qu'une erreur. Faster R-CNN et Mask R-CNN sont un
cas légèrement différent : tous deux ont déjà exécuté une NMS dans le modèle,
à un seuil upstream fixe que `iou` ne peut pas modifier par une méthode prise
en charge.

Les familles suivantes utilisent bien ce paramètre : YOLOv1 à YOLOv4, YOLOv7,
YOLOv9, YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet et SSD.

Trois options de prédiction rendent `iou` pertinent même pour un prédicteur
d'ensembles, car toutes trois agissent sur des bounding boxes après la fin du
modèle :

- `tiling=True` réconcilie les tuiles qui se chevauchent avec une NMS par classe au seuil `iou`
- `augment=True` fusionne les vues retournées avec une NMS par classe au seuil `iou`
- `agnostic_nms=True` supprime les bounding boxes qui se chevauchent entre
  classes au seuil `iou`, décrit [plus bas](#agnostic_nms)

Les deux premières sont décrites dans les
[performances d'inférence](/docs/predict/performance).

Les détecteurs à vocabulaire ouvert suivent leur propre règle. Une famille
dont le processeur exécute une NMS déclare son propre seuil par défaut et
respecte `iou`, comme OMDet-Turbo. Les familles qui n'appliquent aucune
suppression, Grounding DINO, OWLv2 et OV-DEIM, émettent un avertissement
lorsque `iou` est transmis. Cet avertissement est le seul de ce type dans la
bibliothèque.

## max_det

`max_det` limite le nombre de prédictions renvoyées pour une image. Il
s'applique partout, mais avec des mécanismes différents : une famille avec NMS
tronque après la suppression, tandis qu'un prédicteur d'ensembles l'utilise
comme taille de sa sélection top-k.

Certaines familles limitent la valeur en dessous de votre demande parce que
leur configuration upstream de référence l'impose. SSD s'arrête à 200, et FCOS à sa
propre limite de détections par image. Augmenter `max_det` au-delà n'a aucun
effet.

`max_det` est appliqué de façon centrale plutôt que par famille dans deux cas :
l'inférence par tuiles, où la liste fusionnée est tronquée après la
réconciliation des tuiles, et les appels avec `classes` ou `agnostic_nms=True`,
où la coupure intervient après le filtre ou la suppression.

## Filtrage par classe

<code-tabs name="classes" />

`classes` accepte une liste d'identifiants de classe et conserve uniquement les
prédictions dont la classe y figure. Les identifiants indexent `result.names`.
La méthode la plus sûre pour en obtenir un consiste à lire `names` sur un
résultat plutôt qu'à supposer l'ordre du dataset.

Le filtrage intervient de façon centrale après le post-traitement de chaque
famille, dans le passage unique emprunté par toutes les voies de prédiction.
Cela entraîne deux conséquences importantes. Il fonctionne pour chaque
famille, y compris celles sans NMS. Il filtre aussi les charges utiles alignées
sur les bounding boxes. Les masques, points clés et bounding boxes orientées
sont donc réduits avec elles au lieu de rester désalignés.

Sur la ligne de commande, `classes` accepte un entier seul, une liste ou une
chaîne séparée par des virgules :

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Le filtrage n'améliore pas gratuitement l'exactitude. Un modèle consacre
toujours ses ressources à prédire des classes que vous supprimez ensuite. Avec
`classes` défini, la famille conserve un budget de 300 candidats, ou `max_det`
si cette valeur est plus élevée, et la coupure à `max_det` intervient après le
filtre : un petit `max_det` n'évince donc pas votre classe. Les backends
exportés utilisent le même budget, y compris les moteurs TensorRT par batch.
Une image qui contient plus de 300 bounding boxes de classes indésirables au
score plus élevé peut tout de même remplir le budget avant que votre classe ne
soit atteinte ; augmentez `max_det` au-delà de 300 dans ce cas.

## agnostic_nms

La NMS propre à chaque famille tient compte des classes : par défaut, deux
bounding boxes qui se chevauchent mais appartiennent à des classes différentes
survivent donc toutes les deux, quelle que soit la valeur de `iou`.
`agnostic_nms=True` ajoute une étape indépendante des classes : parmi les
bounding boxes qui se chevauchent au-delà de `iou`, seule celle au score le plus
élevé survit, quelles que soient leurs classes.

<code-tabs name="agnostic" />

L'étape s'exécute sur les détections terminées de la famille, après le filtre
`classes` et avant la coupure à `max_det`. Elle se comporte donc de la même façon
sur les prédicteurs d'ensembles qui n'exécutent aucune NMS propre, sur les
backends exportés, et avec `augment=True` et `tiling=True`. Les masques et les
points clés sont supprimés avec leur bounding box, et les bounding boxes
orientées sont comparées par IoU pivoté. Elle couvre les tâches qui renvoient des
bounding boxes : détection, segmentation, pose et OBB.

La famille conserve un budget de 300 candidats, ou `max_det` si cette valeur est
plus élevée, sur lequel l'étape travaille. Avec `max_det` inférieur à 300, des
candidats au score plus faible comblent les places que libère la suppression. À
partir de 300, ce qui inclut la valeur par défaut, ces places restent vides.

Là où elle ne s'applique pas :

- Sur le chemin de prédiction standard, une tâche sans bounding boxes lève
  `ValueError`.
- `LibreEnsemble` et [HRNet](/docs/models/hrnet) lèvent `NotImplementedError`.
- Les modèles `LibreSAM` et les [détecteurs 3D](/docs/tasks/3d-object-detection),
  WildDet3D, 3D-MOOD, FCOS3D et DetAny3D, ne prennent pas cet argument et lèvent
  `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body) et
  [LibreFaceRec](/docs/models/librefacerec) exécutent leur propre chemin de
  prédiction et l'ignorent.

`val(agnostic_nms=True)` applique la même étape à la validation de détection et
de segmentation ; voir [Validation et mesures](/docs/train/validation).

## Arguments refusés par predict

Deux arguments lèvent une erreur au lieu d'un avertissement : `visualize` et
`embed` lèvent tous deux `NotImplementedError`. Pour obtenir des embeddings,
chargez le modèle avec `task="embed"`, puis appelez normalement `predict` ou
`embed`. `agnostic_nms=True` lève une erreur là où il ne s'applique pas, comme
listé ci-dessus.

Tout argument inconnu lève `TypeError` en indiquant les options prises en
charge. Une faute de frappe échoue donc immédiatement au lieu d'être ignorée
silencieusement.

Les arguments suivants sont acceptés, accompagnés d'un avertissement, puis
supprimés : `boxes`, `dnn`, `half`, `line_width`, `retina_masks`,
`show_conf`, `show_labels` et `verbose`.
