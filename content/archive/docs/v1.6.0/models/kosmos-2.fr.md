---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 dans LibreYOLO : détection d''objets avec grounding'
description: >-
  Kosmos-2 dans LibreYOLO : installez, définissez un vocabulaire ouvert et
  prédisez des boîtes ancrées avec le modèle sous licence MIT de Microsoft.
lead: >-
  Kosmos-2 est le modèle de grounding de Microsoft : il légende une image, puis
  localise par une boîte chaque groupe nominal de cette légende. LibreYOLO
  l'enveloppe en détecteur d'objets à vocabulaire ouvert : fournissez la liste
  de classes au moment de la prédiction.
keywords:
  - Kosmos-2
  - modèle vision-langage
  - grounding
  - détection à vocabulaire ouvert
  - détecter des objets sans entraînement
  - Kosmos-2 Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vidéo
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # index de webcam, flux RTSP ou liste .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Installation

Kosmos-2 appartient au niveau « VLM comme détecteur » de LibreYOLO, une surface
produit distincte des familles à base de checkpoints, avec sa propre factory. Il
demande l'extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement. LibreYOLO charge directement le dépôt
`microsoft/kosmos-2-patch14-224` de Microsoft lui-même ; contrairement à
Florence-2, aucun ré-hébergement communautaire n'est nécessaire ici.

<code-tabs name="predict" />

Cette famille se charge via la factory `LibreVLM()`, et non `LibreYOLO()` : les
familles VLM ne déclarent aucun chargeur de checkpoint, si bien que le routage
par suffixe de fichier décrit sur les autres pages de modèles ne s'applique pas
ici. `set_classes()` définit le vocabulaire que Kosmos-2 doit trouver ; il est
rémanent, donc il reste actif pour tous les appels `predict()`/`track()`
suivants jusqu'à ce que vous le redéfinissiez. Kosmos-2 fait le grounding de
groupes nominaux plutôt que de faire correspondre une étiquette à l'identique,
si bien que le wrapper de LibreYOLO accepte une correspondance partielle : une
classe nommée `"boat"` correspond aussi à un groupe généré comme « the boats ».
Chaque détection porte la même confiance de remplissage, si bien que le filtrage
par `conf` est tout ou rien plutôt qu'un classement, et `iou` n'a aucun effet
ici, puisque le wrapper construit la liste des détections directement à partir
des entités ancrées, sans étape de déduplication. `chat()` lève
`NotImplementedError`, car Kosmos-2 est piloté par un prompt `<grounding>` et
non par un template de chat. Le CLI de LibreYOLO ne couvre pas ce niveau : il
n'existe pas de forme `libreyolo predict model=...` pour lui. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Une seule taille : `kosmos-2-patch14-224`, en 224 px, chargée avec
`LibreVLM("kosmos-2")`. C'est un modèle de 2023, et le wrapper de LibreYOLO note
lui-même que son grounding est plus grossier que celui des détecteurs plus
récents de ce niveau.

LibreYOLO n'entraîne, ne valide ni n'exporte Kosmos-2 : `train()`, `val()` et
`export()` lèvent tous `NotImplementedError` pour chaque famille de ce niveau
(voir le niveau de prise en charge ci-dessus). Faites du fine-tuning de Kosmos-2
en amont et chargez les poids obtenus si vous avez besoin d'un vocabulaire
personnalisé figé dans le modèle ; contrôlez la sortie de `predict()` à l'œil
plutôt qu'avec une passe de validation à la COCO, puisque chaque détection porte
la même confiance de remplissage.

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
