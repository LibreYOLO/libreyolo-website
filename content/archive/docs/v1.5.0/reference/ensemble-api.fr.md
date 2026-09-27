---
title: API Ensemble
seo_title: API LibreEnsemble et opérations de fusion
description: "LibreEnsemble, ExternalDetector et les trois opérations de fusion de libreyolo.ops\_: weighted boxes fusion, sa variante à graines et la fusion NMS consciente des classes."
lead: >-
  LibreEnsemble exécute plusieurs détecteurs sur la même image et fusionne leurs
  détections dans un objet Results unique. La fusion intervient après le
  post-traitement propre à chaque membre, chacun conserve donc sa taille
  d'entrée, sa normalisation et sa suppression.
keywords:
  - LibreEnsemble
  - weighted boxes fusion
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - consensus min_votes
last_verified: 1.5.0
verification: >-
  Signatures et valeurs par défaut lues dans libreyolo/ensemble/model.py et
  libreyolo/ops/fusion.py en v1.5.0. Intention de conception lue dans
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Deux membres, fusion par défaut'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Une source image unique renvoie un objet Results, et non une liste.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Consensus et seuils par membre
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: Opération de fusion sans modèle
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| Argument | Valeur par défaut | Signification |
|---|---|---|
| `members` | | Au moins deux détecteurs |
| `weights` | `None` | Facteurs de confiance par membre\u00a0; tous à `1.0` si omis |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"` ou un callable |
| `fusion_iou` | `0.55` | Seuil IoU pour le regroupement de fusion |
| `min_votes` | `1` | Conserver uniquement les bounding boxes confirmées par au moins ce nombre de membres |

Un membre peut être un chemin de poids résolu avec la fabrique `LibreYOLO()`,
un modèle déjà construit, un backend exporté ou un `ExternalDetector`. Chaque
membre doit être un modèle de la tâche detect.

<code-tabs name="usage" />

La construction refuse moins de deux membres, une liste `weights` de longueur
incorrecte, un poids non positif, un `min_votes` qui n'est pas un entier
positif et un `min_votes` supérieur au nombre de membres. `fusion="nms"` avec
`min_votes > 1` lève aussi une erreur, car la NMS supprime l'appartenance aux
groupes et ne peut pas compter les votes.

`weights` pondère la confiance accordée à chaque membre. Un poids plus élevé
attire les coordonnées et scores fusionnés vers ce membre. La convention
consiste à les rendre proportionnels à la mAP de validation.

## Espaces de classes

Les membres dont les `names` sont identiques les transmettent directement.
Sinon, les espaces de classes sont réunis par nom, les identifiants de classes
des membres sont remappés par des tables de correspondance et
`Results.names` fusionné représente l'union. La fusion ne regroupe les bounding
boxes qu'au sein d'une même classe unifiée. Une classe connue d'un seul membre
traverse donc la fusion telle quelle. Une différence produit un avertissement
lors de la construction.

`min_votes` est limité pour chaque classe au nombre d'espaces d'étiquettes de
membres qui contiennent celle-ci. Le consensus reste ainsi pertinent sur des
vocabulaires partiellement partagés.

## Appeler l'ensemble

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` est un alias de `__call__`. La valeur renvoyée est l'objet `Results`
habituel, dont `speed` décompose le coût par membre et ajoute une entrée
`fusion`. Une source image unique renvoie un objet, une liste ou un répertoire
renvoie une liste, et `stream=True` renvoie un générateur.

`conf`, `iou` et `device` sont diffusés à chaque membre et acceptent aussi une
valeur par membre. `conf=[0.25, 0.4]` donne ainsi un seuil de 0.25 au membre 0
et de 0.4 au membre 1. `imgsz` est diffusé s'il s'agit d'un entier ou d'un
tuple et ne s'applique par membre que s'il s'agit d'une liste.
`imgsz=(480, 640)` est donc une taille rectangulaire unique pour tous, tandis
que `imgsz=[480, 640]` donne 480 au membre 0 et 640 au membre 1. Chaque entrée
doit être valide pour la famille concernée.

`augment` est diffusé aux membres qui prennent en charge l'augmentation à
l'inférence, tandis que les backends exportés l'ignorent. `classes` accepte les
identifiants de classes de l'union et `max_det` s'applique au résultat fusionné.
Les membres produisent donc largement et l'ensemble tronque une seule fois.
`batch` est accepté pour assurer la parité de l'API\u00a0; les images sont traitées
séquentiellement.

`val()` et `export()` lèvent `NotImplementedError`. Validez et exportez chaque
membre séparément.

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Adapte tout callable de détection en membre. `fn` accepte une image PIL et
renvoie `(boxes, scores, labels)`, où les bounding boxes sont des xyxy dans les
pixels de l'image d'origine et les étiquettes des identifiants de classes
valides dans `names`. Les tenseurs, tableaux et listes imbriquées fonctionnent
tous. LibreYOLO n'importe rien depuis le code externe.

L'adaptateur valide la valeur renvoyée\u00a0: elle doit être un tuple à 3 éléments,
les bounding boxes doivent avoir la forme `(N, 4)`, les trois tableaux doivent
avoir la même longueur et chaque identifiant de classe doit figurer dans
`names`. Les détections dont la confiance est inférieure ou égale à `conf`
sont supprimées avant la fusion.

## Opérations de fusion

Les primitives de fusion sont des opérations torch autonomes dans
`libreyolo.ops`. Elles sont indépendantes des modèles et peuvent être importées
seules, d'où leur export séparé de l'ensemble.

<code-tabs name="ops" />

Toutes trois acceptent les mêmes arguments positionnels, `boxes, scores,
labels, model_ids`, et renvoient `(boxes, scores, labels)`.

| Opération | Clé du registre | Comportement |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Weighted boxes fusion séquentielle et fidèle à l'article |
| `wbf_seeded` | `wbf_seeded` | Variante parallèle en une passe de la même réduction |
| `nms_fusion` | `nms` | Concatène tous les éléments et applique une NMS consciente des classes |

`FUSIONS` associe les trois clés du registre aux callables et `LibreEnsemble`
y recherche `fusion=`.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` possède une signature identique. `nms_fusion` accepte les mêmes
arguments sauf `conf_type` et lève `ValueError` lorsque `min_votes > 1`.

Dans `weighted_boxes_fusion`, les détections sont parcourues par ordre
décroissant de confiance pondérée. Chacune rejoint soit le groupe existant
dont la bounding box fusionnée courante présente le meilleur chevauchement, à
une IoU supérieure à `iou_thr` et pour la même étiquette, soit un nouveau
groupe. La bounding box fusionnée d'un groupe est la moyenne de ses
coordonnées pondérée par la confiance. Son score est la moyenne pondérée ou le
maximum des confiances, remis à l'échelle afin que les bounding boxes
confirmées par moins de modèles obtiennent un score inférieur.

`wbf_seeded` choisit les graines des groupes avec une NMS consciente des
classes au seuil `iou_thr`, affecte chaque détection à la graine de même
étiquette présentant la meilleure IoU, puis réduit chaque groupe de la même
manière. Les formes des groupes ne changent jamais au cours de la passe,
l'opération entière repose donc sur des calculs tensoriels de forme fixe. Les
deux variantes concordent lorsque les groupes sont sans ambiguïté, mais peuvent
différer légèrement sur des chaînes de groupes qui se chevauchent.

`nms_fusion` conserve sans modification la bounding box de plus forte
confiance de chaque groupe qui se chevauche. Les `weights` par modèle mettent
à l'échelle les confiances uniquement pour le classement de suppression, et
les bounding boxes survivantes conservent leurs scores d'origine.

## Fusion personnalisée

`fusion=` accepte aussi un callable ayant la même signature que les opérations
ci-dessus. Son nom est consigné dans `ens.fusion`, ou `"custom"` s'il n'en
possède aucun. La valeur renvoyée est validée\u00a0: il doit s'agir d'un triplet
`(boxes, scores, labels)` aux formes cohérentes.
