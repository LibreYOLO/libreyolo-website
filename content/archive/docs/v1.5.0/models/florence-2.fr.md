---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 dans LibreYOLO : détection à vocabulaire ouvert'
description: >-
  Florence-2 dans LibreYOLO : installez, définissez un vocabulaire ouvert et
  prédisez des boîtes avec le modèle de vision de Microsoft sous licence MIT.
lead: >-
  Florence-2 est le modèle de fondation visuel de Microsoft, piloté par un token
  de tâche plutôt que par une tête de détection figée. LibreYOLO l'enveloppe en
  détecteur d'objets à vocabulaire ouvert : fournissez la liste de classes au
  moment de la prédiction.
keywords:
  - Florence-2
  - modèle vision-langage
  - détection à vocabulaire ouvert
  - VLM détection d'objets
  - grounding
  - Florence-2 Microsoft
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vidéo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # index de webcam, flux RTSP ou liste .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Installation

Florence-2 appartient au niveau « VLM comme détecteur » de LibreYOLO, une
surface produit distincte des familles à base de checkpoints, avec sa propre
factory. Il demande l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement. LibreYOLO télécharge le checkpoint réhébergé par
florence-community plutôt que le dépôt d'origine `microsoft/Florence-2-*` ;
voir la section Licence pour savoir pourquoi.

<code-tabs name="predict" />

Cette famille se charge via la factory `LibreVLM()`, et non `LibreYOLO()` : les
familles VLM ne déclarent aucun chargeur de checkpoint, si bien que le routage
par suffixe de fichier décrit sur les autres pages de modèles ne s'applique pas
ici. `set_classes()` définit le vocabulaire que Florence-2 doit chercher dans
l'image ; il est rémanent, donc il reste actif pour tous les appels
`predict()`/`track()` suivants jusqu'à ce que vous le redéfinissiez. L'objet
`Results` renvoyé porte des `boxes` de la même forme que dans n'importe quelle
autre famille, mais chaque détection porte la même confiance de remplissage, si
bien que le filtrage par `conf` est tout ou rien plutôt qu'un classement, et
`iou` n'a aucun effet : le wrapper de Florence-2 construit la liste des
détections directement à partir de la sortie analysée du token de tâche, sans
étape de déduplication. `chat()` lève ici `NotImplementedError`, car Florence-2
est piloté par le token de tâche `<OPEN_VOCABULARY_DETECTION>` et non par un
template de chat. Le CLI de LibreYOLO ne couvre pas ce niveau : il n'existe pas
de forme `libreyolo predict model=...` pour lui. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Deux tailles : Florence-2-base et Florence-2-large, toutes deux en 768 px,
chargées avec `LibreVLM("florence-2-base")` ou `LibreVLM("florence-2-large")`.
LibreYOLO n'a pas publié de benchmark comparant leur exactitude.

LibreYOLO n'entraîne, ne valide ni n'exporte Florence-2 : `train()`, `val()` et
`export()` lèvent tous `NotImplementedError` pour chaque famille de ce niveau
(voir le niveau de prise en charge ci-dessus). Faites du fine-tuning de
Florence-2 en amont et chargez les poids obtenus si vous avez besoin d'un
vocabulaire personnalisé figé dans le modèle ; contrôlez la sortie de
`predict()` à l'œil plutôt qu'avec une passe de validation à la COCO, puisque
chaque détection porte la même confiance de remplissage.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
