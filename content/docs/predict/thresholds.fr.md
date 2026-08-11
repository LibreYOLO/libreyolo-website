---
title: Seuils et filtrage
seo_title: 'conf, iou et max_det dans LibreYOLO'
description: >-
  Ce que font réellement conf, iou, max_det et classes lors de la prédiction,
  les familles qui ignorent iou faute de NMS, et pourquoi agnostic_nms est sans
  effet.
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
last_verified: 1.5.0
verification: >-
  Valeurs par défaut citées depuis InferenceRunner.__call__ dans
  libreyolo/models/base/inference.py. Comportement NMS par famille lu dans
  chaque module de libreyolo/postprocess/ et recoupé avec _is_nms_free_family
  dans libreyolo/backends/base.py. Filtrage des classes lu dans
  InferenceRunner._apply_classes_filter et _wrap_results. État de agnostic_nms
  lu dans NOOP_PREDICT_KWARGS dans libreyolo/utils/predict_args.py. Gestion du
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
source_hash: 0b978963c356027d
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

La validation utilise délibérément d'autres valeurs par défaut\u00a0: `val()`
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

## iou

`iou` est le chevauchement au-delà duquel la suppression non maximale retire
la bounding box au score le plus faible parmi deux bounding boxes de même
classe. Il n'a de sens que si la famille exécute effectivement une suppression.

Un prédicteur d'ensembles décode un nombre fixe de requêtes et conserve celles
dont le score est le plus élevé. Les doublons sont supprimés dans l'architecture
pendant l'entraînement et non lors d'une étape de post-traitement, aucun seuil
ne peut donc être réglé. Les familles suivantes acceptent `iou` pour assurer
la parité de l'API, mais l'ignorent\u00a0:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR et la tête de bout en bout
de YOLOv9. Les variantes construites sur ces décodeurs héritent de ce
comportement.

<code-tabs name="nmsfree" />

La plupart l'indiquent dans les docstrings de leur post-traitement, mais aucun
avertissement n'est levé à l'exécution. Un balayage de `iou` sur RF-DETR produit
donc une ligne plate plutôt qu'une erreur. Faster R-CNN et Mask R-CNN sont un
cas légèrement différent\u00a0: tous deux ont déjà exécuté une NMS dans le modèle,
à un seuil upstream fixe que `iou` ne peut pas modifier par une méthode prise
en charge.

Les familles suivantes utilisent bien ce paramètre\u00a0: YOLOv1 à YOLOv4, YOLOv7,
YOLOv9, YOLOX, YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet et SSD.

Deux options de prédiction rendent `iou` pertinent même pour un prédicteur
d'ensembles, car elles fusionnent toutes deux des bounding boxes après la fin
du modèle\u00a0:

- `tiling=True` réconcilie les tuiles qui se chevauchent avec une NMS par classe au seuil `iou`
- `augment=True` fusionne les vues retournées avec une NMS par classe au seuil `iou`

Toutes deux sont décrites dans les
[performances d'inférence](/docs/predict/performance).

Les détecteurs à vocabulaire ouvert suivent leur propre règle. Une famille
dont le processeur exécute une NMS déclare son propre seuil par défaut et
respecte `iou`, comme OMDet-Turbo. Les familles qui n'appliquent aucune
suppression, Grounding DINO, OWLv2 et OV-DEIM, émettent un avertissement
lorsque `iou` est transmis. Cet avertissement est le seul de ce type dans la
bibliothèque.

## max_det

`max_det` limite le nombre de prédictions renvoyées pour une image. Il
s'applique partout, mais avec des mécanismes différents\u00a0: une famille avec NMS
tronque après la suppression, tandis qu'un prédicteur d'ensembles l'utilise
comme taille de sa sélection top-k.

Certaines familles limitent la valeur en dessous de votre demande parce que
leur configuration upstream de référence l'impose. SSD s'arrête à 200, la
segmentation d'instances de RTMDet à 100 et FCOS à sa propre limite de
détections par image. Augmenter `max_det` au-delà n'a aucun effet.

L'inférence par tuiles est le seul endroit où `max_det` est appliqué de façon
centrale plutôt que par famille. La liste fusionnée y est tronquée après la
réconciliation des tuiles.

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
chaîne séparée par des virgules\u00a0:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Le filtrage n'améliore pas gratuitement l'exactitude. Un modèle consacre
toujours ses ressources à prédire des classes que vous supprimez ensuite, et
la famille applique `max_det` avant le filtre. Une image encombrée de classes
indésirables peut donc atteindre la limite avant votre classe. Réduisez `conf`
ou augmentez `max_det` dans ce cas.

## agnostic_nms

`agnostic_nms` est accepté mais ne fait rien. Le transmettre produit un
avertissement indiquant qu'il est sans effet et ne sert qu'à la compatibilité
avec la ligne de commande, puis l'argument est supprimé.

Aucun mode de suppression indépendant des classes n'existe. Chaque appel NMS
de la bibliothèque tient compte des classes. Deux bounding boxes qui se
chevauchent mais appartiennent à des classes différentes sont donc conservées,
quelle que soit la valeur de `iou`. Si cela pose problème, filtrez d'abord avec
`classes` ou appliquez vous-même une suppression entre classes sur
`result.boxes`.

## Arguments refusés par predict

Deux arguments lèvent une erreur au lieu d'un avertissement\u00a0: `visualize` et
`embed` lèvent tous deux `NotImplementedError`. Pour obtenir des embeddings,
chargez le modèle avec `task="embed"`, puis appelez normalement `predict` ou
`embed`.

Tout argument inconnu lève `TypeError` en indiquant les options prises en
charge. Une faute de frappe échoue donc immédiatement au lieu d'être ignorée
silencieusement.

Les arguments suivants sont acceptés, accompagnés d'un avertissement, puis
supprimés\u00a0: `agnostic_nms`, `boxes`, `dnn`, `half`, `line_width`,
`retina_masks`, `show_conf`, `show_labels` et `verbose`.
