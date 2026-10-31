---
title: InternVL3
families:
  - internvl3
seo_title: "InternVL3\_: la détection à vocabulaire ouvert dans LibreYOLO"
description: "Utilisez InternVL3 dans LibreYOLO pour la détection d'objets à vocabulaire ouvert. Prédisez avec n'importe quelle étiquette textuelle\_; l'entraînement, la validation et l'export ne sont pas pris en charge."
lead: "InternVL3 est un grand modèle de langage multimodal natif publié par OpenGVLab, qui apprend conjointement la vision et le langage en une seule phase de pré-entraînement. LibreYOLO l'encapsule comme détecteur d'objets à vocabulaire ouvert\_: n'importe quelle liste d'étiquettes textuelles devient l'ensemble des classes, sans tête fixe et sans fine-tuning à faire."
keywords:
  - InternVL3
  - InternVL
  - modèle vision-langage
  - détection à vocabulaire ouvert
  - VLM
  - OpenGVLab
  - détecter des objets avec du texte
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Vocabulaire ouvert, n'importe quels mots, pas de tête de classes
        # fixe. Persiste sur chaque predict()/track() jusqu'au prochain appel.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat brut
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # La porte de sortie sous la couche de détection, pour les questions
        # libres, le comptage, ou tout prompt que le wrapper boxes ignore.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Installation

InternVL3 a besoin de l'extra `vlm`, qui embarque `transformers` pour le backbone de chat-template.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

`LibreInternVL3` est une classe Python, pas un checkpoint `.pt` : elle ne se charge pas via la factory `LibreYOLO()`, et la CLI `libreyolo` ne la résout pas. La factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) atteint elle aussi cette famille par alias, par exemple `LibreVLM("internvl3-2b")` ; la classe utilisée ci-dessous est celle qu'elle construit. Les poids viennent des dépôts Hugging Face `-hf` d'OpenGVLab, pas d'un miroir LibreYOLO ; le premier appel les télécharge et les met en cache localement, après avoir journalisé un avis de licence unique pour les poids Qwen, dont l'accès est restreint.

<code-tabs name="predict" />

`result.boxes` porte les détections parsées comme dans n'importe quelle autre famille. La confiance est une valeur de remplissage : InternVL3 n'émet aucun score par boîte, donc chaque détection reçoit la même confiance constante, et `conf=` ne fait qu'écarter les lignes situées sous cette constante, il ne les classe pas. `iou` élimine les boîtes quasi dupliquées de la même classe au-dessus du recouvrement indiqué, un effet de bord du décodage glouton qui répète un objet ; ce n'est pas une passe de NMS par classe. Sautez `set_classes()` et le vocabulaire retombe sur les noms COCO-80. Voir [la prédiction](/docs/predict) pour les sources, le streaming et le traitement des résultats.

## Variantes

Trois tailles : 1b, 2b et 8b, toutes des checkpoints `-hf` natifs d'OpenGVLab (un backbone LLM Qwen, et non l'architecture à deux tours que décrit l'article original d'InternVL). Le harness de benchmarks de LibreYOLO n'a pas mesuré cette famille, il n'y a donc pas de chiffres d'exactitude publiés pour les comparer ; choisissez une taille en fonction de votre propre budget de calcul.

LibreYOLO n'expose cette famille que pour la prédiction. `train()`, `val()` et `export()` lèvent tous `NotImplementedError` : faites le fine-tuning en amont et chargez le résultat, la validation sur dataset est ignorée parce qu'une confiance de remplissage rendrait le mAP COCO trompeur, et l'export sort du périmètre pour un modèle génératif sans state dict à tracer.

## Licence

<provenance-box>

Le code propre à InternVL3 est sous MIT, permissif et utilisable dans des produits commerciaux et à code fermé. Les checkpoints `-hf` que charge cette famille portent un backbone LLM Qwen et sont sous une licence distincte, la Qwen License d'Alibaba Cloud : libres d'usage, de modification et de redistribution moyennant une attribution « Built with Qwen » ou « Improved using Qwen », et avec un plafond de 100 millions d'utilisateurs actifs mensuels sur l'usage commercial, au-delà duquel l'autorisation d'Alibaba elle-même est requise. LibreYOLO n'héberge ni ne redistribue ces poids : `LibreInternVL3` télécharge la taille correspondante directement depuis `OpenGVLab/InternVL3-<size>-hf` sur Hugging Face au premier lancement, et journalise un avis unique pour la Qwen License avant ce téléchargement.

</provenance-box>

## Citation

<citation-block />
