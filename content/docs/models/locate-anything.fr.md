---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything : détection à vocabulaire ouvert et pointage'
description: >-
  Utilisez LocateAnything dans LibreYOLO pour la détection à vocabulaire ouvert
  et le pointage. Prédisez avec n'importe quelle étiquette texte ;
  l'entraînement, la validation et l'export ne sont pas pris en charge.
lead: >-
  LocateAnything est un modèle de grounding vision-langage publié par NVIDIA qui
  décode les boîtes englobantes et les points en parallèle plutôt qu'un token de
  coordonnée à la fois. LibreYOLO l'intègre comme détecteur et pointeur à
  vocabulaire ouvert : n'importe quelle liste d'étiquettes texte devient
  l'ensemble de classes, sans tête figée et sans fine-tuning requis.
keywords:
  - LocateAnything
  - NVIDIA
  - modèle vision-langage
  - détection à vocabulaire ouvert
  - détecter des objets à partir de texte
  - pointer un objet dans une image
  - VLM python
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Vocabulaire ouvert : n'importe quels mots, pas de tête figée.
        # Persiste sur les predict()/track() suivants jusqu'à redéfinition.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt de pointage
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" renvoie un point par objet trouvé au lieu d'une boîte.
        # Changez de tâche sur un modèle chargé via model.set_task("point").
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Chat brut
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # La trappe de sortie sous la couche de détection : questions
        # libres, comptage, ou tout prompt que le wrapper ne couvre pas.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Installation

LocateAnything a besoin de l'extra `vlm`, qui installe `transformers` ainsi que
les paquets `decord`, `lmdb` et `peft` que son code distant Hugging Face
importe au chargement.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

`LibreLocateAnything` est une classe Python, pas un checkpoint `.pt` : elle ne
se charge pas via la factory `LibreYOLO()`, et la CLI `libreyolo` ne la résout
pas. La factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) atteint elle
aussi cette famille par alias, par exemple `LibreVLM("locate-anything")` ; la
classe utilisée ci-dessous est celle qu'elle construit. Son chargement
télécharge et exécute le code de modèle distant de NVIDIA depuis Hugging Face,
si bien que LibreYOLO fige le téléchargement sur une révision de commit fixe
plutôt que sur la branche mutable `main`, et journalise un avis de licence
unique avant le premier téléchargement.

<code-tabs name="predict" />

`result.boxes` (tâche `detect`) et `result.points` (tâche `point`) portent la
sortie analysée comme dans toute autre famille. La confiance est un
substitut : LocateAnything n'émet aucun score par boîte, donc chaque détection
reçoit la même confiance constante, et `conf=` ne fait qu'écarter les lignes
situées sous cette constante, il ne les classe pas. Omettez `set_classes()` et
le vocabulaire retombe sur les noms COCO-80. Voir [la
prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Variantes

Une seule taille publiée, 3b. Deux tâches partagent les mêmes poids : `detect`
(la valeur par défaut) renvoie des boîtes, et `task="point"` renvoie à la place
un unique point par objet trouvé, dans `result.points` ; passez de l'une à
l'autre sur un modèle déjà chargé avec `model.set_task("point")`. Le harnais de
benchmark de LibreYOLO n'a pas mesuré cette famille, il n'y a donc aucun
chiffre d'exactitude publié auquel se comparer.

LibreYOLO expose cette famille en prédiction seule. `train()`, `val()` et
`export()` lèvent tous `NotImplementedError` : faites le fine-tuning en amont
et chargez le résultat à la place, la validation sur dataset est écartée parce
qu'une confiance de substitution rendrait la mAP COCO trompeuse, et l'export
est hors périmètre pour un modèle génératif dont aucun state dict n'est là pour
être tracé.

## Licence

<provenance-box>

La NVIDIA License autorise l'utilisation, la reproduction et la modification,
mais restreint le modèle et tout dérivé à un usage non commercial, de recherche
ou d'évaluation uniquement, pour quiconque autre que NVIDIA et ses affiliés :
il n'y a ni seuil de revenus ni exception payante. LocateAnything-3B compose
par ailleurs deux autres composants sous licence : un backbone de langage
Qwen2.5-3B-Instruct sous la Qwen Research License, et un encodeur visuel
MoonViT-SO-400M sous MIT. LibreYOLO n'héberge, ne met en miroir ni ne
redistribue rien de tout cela : `LibreLocateAnything` télécharge les poids et
le code distant requis directement depuis `nvidia/LocateAnything-3B` sur
Hugging Face, figé sur un commit fixe, lors de sa première exécution.

</provenance-box>

## Citation

<citation-block />
