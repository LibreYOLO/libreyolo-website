---
title: API vision-langage
seo_title: "API LibreVLM\_: alias, set_classes et chat"
description: >-
  La fabrique LibreVLM, tous les alias de modèles, le vocabulaire persistant de
  set_classes, set_task, la porte de sortie chat et la raison pour laquelle la
  confiance est factice.
lead: >-
  LibreVLM charge un modèle vision-langage génératif et l'utilise comme
  détecteur d'objets. La liste de classes sert de prompt plutôt que de tête
  fixe, et le modèle renvoie les mêmes Results que toute autre famille.
keywords:
  - LibreVLM
  - modèle vision langage détection
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: "Alias lus dans libreyolo/models/vlm/__init__.py\_; dépôts, tailles et listes de tâches lus dans les modules de familles sous libreyolo/models/vlm/ ainsi que dans libreyolo/models/sensenova/model.py\_; règles d'appel et erreurs lues dans libreyolo/models/vlm/base.py, le tout en v1.5.0."
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Détecter dans un vocabulaire ouvert
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Poser une question libre
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Installer

Ce niveau nécessite l'extra `vlm`.

<code-tabs name="install" />

## La fabrique

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` est un alias, et non un chemin. `**kwargs` atteint le constructeur de
la famille, qui accepte `device`, `names` (le vocabulaire initial, équivalent
à un appel à `set_classes` après le chargement), `prompt` (pour remplacer le
prompt de détection) et `max_new_tokens`. Un alias inconnu lève `ValueError`
en énumérant tous les alias.

<code-tabs name="usage" />

## Alias

| Famille | Alias | Tailles | Poids |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot upstream épinglé |

L'alias par défaut est `qwen3-vl-4b`. Les tailles des alias par défaut de
chaque famille sont celles qui apparaissent en premier\u00a0: `qwen3-vl` se résout
en `4b`, `lfm2-vl` en `450m`, `internvl3` en `2b`, `smolvlm2` en `2.2b` et
`florence-2` en `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` et `LibreMODUS`
(également orthographié `LibreModus`) sont exportés au niveau du package.

## Tâches

La plupart des familles proposent uniquement `detect`. Deux en proposent
davantage\u00a0:

| Famille | Tâches prises en charge |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Comme la tâche est déterminée par le prompt et non intégrée à un checkpoint,
elle peut être modifiée sur un modèle déjà chargé\u00a0:

```python
model.set_task(task: str) -> LibreVLMModel
```

La tâche est validée par rapport à la liste prise en charge par la famille,
reste appliquée aux appels `predict()` et `track()` ultérieurs, et le modèle
est renvoyé afin de pouvoir chaîner les appels.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Définit le vocabulaire ouvert. Tous les mots fonctionnent, car ils sont
transmis au modèle sous forme de prompt au lieu d'être limités à une tête fixe.
La liste ne doit pas être vide et ses entrées doivent être uniques sans tenir
compte de la casse. Une chaîne seule lève `TypeError`, car elle serait
décomposée en classes d'un caractère. Le vocabulaire est persistant\u00a0:
définissez-le une fois après le chargement, il reste actif jusqu'à sa prochaine
définition.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Génération multimodale brute\u00a0: une image et un prompt en entrée, le texte
décodé verbatim en sortie. C'est la porte de sortie sous l'outil pratique de
détection, destinée aux questions libres, au comptage ou à un format de sortie
non couvert par le wrapper de détection. `max_new_tokens` utilise par défaut
la valeur `MAX_NEW_TOKENS` de la famille, soit 1024 sur la classe de base. Le
décodage est glouton avec une légère pénalité de répétition.

## Confiance

La sortie générée ne possède aucune confiance calibrée par bounding box. Cette
version attribue une valeur factice constante afin que `predict`, le dessin et
`track` fonctionnent. Le filtrage `conf=` et la mAP deviennent donc
approximatifs plutôt que significatifs. C'est également la raison pour laquelle
`val()` lève une erreur\u00a0: une mAP COCO calculée sur des scores factices serait
trompeuse.

## Prédire et suivre

L'interface de prédiction standard s'applique et `track()` fonctionne. Un
détecteur VLM s'intègre donc au même pipeline que toute autre famille. Deux
politiques de classe diffèrent d'un détecteur convolutionnel\u00a0: l'augmentation
à l'inférence est désactivée, car une augmentation multi-échelle n'a aucun sens
pour un générateur à résolution fixe, et la prédiction par batch est
désactivée, car la génération est autorégressive et le prétraitement renvoie
un encodage texte-image plutôt qu'un tenseur d'images empilable.

## Fonctionnalités non prises en charge

`train()`, `val()` et `export()` lèvent `NotImplementedError`. Effectuez le
fine-tuning upstream et chargez les poids obtenus.

## Code distant

Chaque famille fournie se charge par l'intermédiaire d'une classe de modèle
native. LibreYOLO n'exécute donc pas de code de dépôt tiers par défaut. Une
famille qui en a réellement besoin doit l'activer explicitement et épingler la
révision d'un snapshot. LocateAnything est la seule à le faire, avec le commit
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS constitue une exception explicite au schéma de checkpoint\u00a0: son
alias se résout en un répertoire de fichiers upstream épinglés plutôt qu'en un
fichier `.pt` LibreYOLO. LibreYOLO ne lui ajoute pas de métadonnées v1.0 et ne
le republie pas.

