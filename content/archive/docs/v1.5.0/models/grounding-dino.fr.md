---
title: Grounding DINO
families:
  - grounding_dino
seo_title: "Grounding DINO dans LibreYOLO\_: la détection open-set"
description: >-
  Utilisez Grounding DINO dans LibreYOLO pour détecter n'importe quel objet
  décrit par du texte. Installez l'extra openvocab et prédisez avec un
  vocabulaire en texte libre.
lead: >-
  Grounding DINO est un détecteur d'objets open-set, développé par IDEA
  Research, qui évalue une image face à un prompt en texte libre plutôt que face
  à une liste de classes figée. LibreYOLO l'intègre comme une famille en
  prédiction seule dans son niveau des détecteurs à vocabulaire ouvert.
keywords:
  - Grounding DINO
  - détection d'objets à vocabulaire ouvert
  - détection open-set
  - détection zero-shot
  - détecter des objets à partir de texte
  - grounding dino python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Seuil de texte
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtre par score de boîte, text_threshold par le score de
        # token de la phrase décodée. Les deux valent 0.25 par défaut.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Installation

Grounding DINO se charge via le niveau des détecteurs à vocabulaire ouvert de
LibreYOLO, qui a besoin de l'extra `openvocab` :

```bash
pip install "libreyolo[openvocab]"
```

Cet extra installe `transformers` et `timm`, les bibliothèques Hugging Face que
ce niveau appelle.

## Prédire

Grounding DINO n'est pas un checkpoint que LibreYOLO charge via `LibreYOLO()`.
Il se charge via la factory sœur `LibreOpenVocab`, qui télécharge un snapshot
Hugging Face au premier usage et le met en cache dans `weights/`.

<code-tabs name="predict" />

`set_classes()` définit un vocabulaire de texte persistant : rappelez-la pour
remplacer la liste, ou omettez-la pour conserver les étiquettes COCO-80 par
défaut. Grounding DINO décode des phrases en forme libre depuis sa propre
sortie texte et les remappe lui-même sur ce vocabulaire, une correspondance
normalisée exacte l'emporte, une correspondance sur un token entier est
acceptée, et une phrase ambiguë ou sans correspondance est écartée plutôt que
devinée, si bien que `school bus` n'est jamais associé à `bus` ou `school`
seuls. Un vocabulaire assez long pour dépasser la limite de tokens de
l'encodeur de texte est découpé en plusieurs prompts, exécutés comme autant de
passes avant distinctes, puis fusionnés en un seul ensemble de détections
plafonné par `max_det`.

`iou` est accepté par compatibilité d'API mais émet un avertissement et ne fait
rien, puisque rien ici n'exécute de non-maximum suppression. `imgsz` et
`augment=True` sont refusés d'emblée : le processeur `transformers` gère le
redimensionnement, et l'augmentation au moment du test est hors périmètre pour
ce niveau. `predict()` sur une seule image renvoie un unique `Results`, pas une
liste ; passez un répertoire, une liste d'images, ou `stream=True` pour une
source vidéo afin d'en obtenir plusieurs. Il n'y a pas de chemin CLI pour cette
famille, `libreyolo predict` ne charge que les checkpoints `.pt` via
`LibreYOLO()`, donc les familles `LibreOpenVocab` s'utilisent depuis Python.
Voir [la prédiction](/docs/predict) pour les types de sources et le streaming.

## Variantes

Deux checkpoints, `t` et `b`. `t` est la taille par défaut de ce niveau quand
aucune n'est indiquée. Tous deux reprennent la version officielle d'IDEA
Research via le `GroundingDinoForObjectDetection` de `transformers`, téléchargé
une fois dans un snapshot Hugging Face hébergé par LibreYOLO qui préserve les
fichiers d'origine. Aucun chiffre d'exactitude ou de latence n'est encore
publié pour cette famille.

L'entraînement, la validation sur dataset et l'export sont tous hors périmètre
pour ce niveau : `train()`, `val()` et `export()` lèvent tous
`NotImplementedError` sans condition. Il s'agit d'un wrapper en prédiction
seule autour d'un checkpoint publié.

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
