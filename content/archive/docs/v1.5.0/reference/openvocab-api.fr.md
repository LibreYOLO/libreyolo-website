---
title: API à vocabulaire ouvert
seo_title: "API LibreOpenVocab\_: alias et arguments"
description: >-
  La fabrique LibreOpenVocab, ses quatre familles et tous leurs alias,
  set_classes, les valeurs conf par famille et les règles de text_threshold et
  iou.
lead: >-
  LibreOpenVocab est la fabrique des détecteurs conditionnés par du texte. La
  liste de classes sert de prompt plutôt que de tête fixe, le vocabulaire est
  donc défini par set_classes et le modèle renvoie des Results de détection
  ordinaires par rapport à celui-ci.
keywords:
  - LibreOpenVocab
  - détection vocabulaire ouvert
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: "Alias lus dans libreyolo/models/openvocab/__init__.py\_; dépôts, tailles et seuils lus dans grounding_dino.py, owlv2.py, omdet_turbo.py et ov_deim.py\_; règles d'appel lues dans libreyolo/models/openvocab/base.py, le tout en v1.5.0. Intention de conception lue dans docs/adr/0008-open-vocab-detector-contract.md."
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Installer

Ce niveau nécessite l'extra `openvocab`.

<code-tabs name="install" />

## La fabrique

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` est un alias, et non un chemin. Les underscores sont convertis en
traits d'union avant la recherche. Les noms qualifiés par famille affichés
dans l'inventaire du CLI, comme `omdet_turbo-t` et `grounding_dino-t`, se
chargent donc tels quels. Un alias inconnu lève `ValueError` en énumérant tous
les alias connus.

Le constructeur accepte `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` et `text_threshold=None`. Transmettre `names`
revient à appeler `set_classes` juste après le chargement. Transmettre
`text_threshold` à une famille qui ne le prend pas en charge lève `TypeError`.

<code-tabs name="usage" />

## Familles et alias

| Famille | Alias | Tailles | Poids |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

L'alias par défaut est `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` et `LibreOMDetTurbo` sont exportés au niveau
du package et peuvent être construits directement avec `size=`. OV-DEIM est
accessible par les alias de la fabrique ci-dessus.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Définit le vocabulaire de tous les appels `predict()` ultérieurs et renvoie le
modèle afin de pouvoir chaîner les appels. La liste ne doit pas être vide, ne
doit contenir que des chaînes et ses entrées doivent être uniques sans tenir
compte de la casse\u00a0; les étiquettes vides sont refusées. Une chaîne seule lève
`TypeError`, car elle serait décomposée en classes d'un caractère.

Après l'appel, `model.names` associe `0..N-1` aux étiquettes dans l'ordre donné,
et `model.nb_classes` vaut `N`.

## Arguments d'appel

Ce niveau réutilise l'interface de prédiction standard avec trois différences.

La valeur par défaut de `conf` est celle de la famille et non la valeur
partagée de 0.25\u00a0:

| Famille | conf par défaut | Suppression |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Son propre post-traitement, seuil 0.5, respecte `iou=` |
| OV-DEIM | 0.25 | Appariement un-à-un avec sélection top-K, sans suppression |

`iou=` n'a de sens que pour une famille qui exécute une suppression.
OMDet-Turbo reçoit le seuil comme argument et utilise 0.5 par défaut lorsque
`iou=` n'est pas défini. Les trois autres n'appliquent aucune suppression,
transmettre `iou=` produit donc un avertissement et sa valeur est ignorée.

`text_threshold=` est propre à Grounding DINO, où sa valeur par défaut est
0.25. Vous pouvez le transmettre au constructeur pour rendre la valeur
persistante, ou à chaque appel. Une valeur par appel ne peut pas être associée
à `stream=True`, car les résultats du streaming sont produits de façon
différée\u00a0; définissez-la plutôt sur le constructeur. Toutes les autres
familles lèvent `TypeError` pour ce paramètre.

`imgsz=` lève `ValueError`\u00a0: le pipeline de prétraitement contrôle le
redimensionnement de ce niveau. `augment=True` lève également une erreur,
l'augmentation à l'inférence étant hors périmètre ici. Les tailles d'entrée
sont consignées par famille uniquement comme référence\u00a0: Grounding DINO 800,
OWLv2 960 et 1008, OMDet-Turbo 640, OV-DEIM 640.

## Fonctionnalités non prises en charge

`train()`, `val()`, `track()` et `export()` lèvent tous
`NotImplementedError`. Effectuez le fine-tuning upstream et chargez les poids
obtenus\u00a0; exécutez `predict()` sur chaque image à la place du suivi. La
validation demanderait un validateur dédié, car le validateur de détection
partagé appelle le modèle avec des tenseurs d'images alors que ce niveau exige
des entrées conditionnées par du texte.

