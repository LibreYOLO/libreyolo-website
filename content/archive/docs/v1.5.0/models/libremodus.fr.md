---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS dans LibreYOLO : analyse d''image any-to-any'
description: >-
  Utilisez LibreMODUS dans LibreYOLO pour la profondeur, les normales, les
  contours et la détection, et pour les composer avec any2any(). Inférence seule
  ; les poids se chargent depuis EPFL-VILAB.
lead: >-
  LibreMODUS est une intégration en inférence seule du checkpoint MODUS 14B-A7B,
  un modèle any-to-any qui transforme une entrée dérivée d'une image en une
  autre : RGB en entrée, profondeur en sortie ; profondeur en entrée, normales
  en sortie ; l'un de ces éléments plus une expression, des boîtes en sortie.
  LibreYOLO prend en charge quatre tâches via l'API predict standard, et un
  ensemble plus large via any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - estimation de profondeur python
  - normales de surface
  - détection de contours
  - grounding textuel
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Sans vocabulaire personnalisé, detect décode les tokens de label
        # COCO du checkpoint en ids de classes COCO-80 contigus.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding textuel
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() bascule la détection en grounding textuel : chaque
        # expression tourne seule et revient par le même contrat Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # Une à trois entrées dérivées d'une image (rgb, depth, normal,
        # canny/edge), plus un texte auxiliaire optionnel, vers une cible.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # Le grounding via any2any() exige une entrée text nommant l'expression.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Installer

LibreMODUS a besoin de son propre extra, qui installe `accelerate` pour le dispatch de grands modèles que ce checkpoint exige.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO ne redistribue pas les poids MODUS et n'en met aucun miroir à disposition. Par défaut, le chargement d'un modèle `LibreMODUS` télécharge les fichiers nécessaires directement depuis `EPFL-VILAB/MODUS`, à une révision Hugging Face épinglée, et un nouveau téléchargement passe toujours par le compte Hugging Face authentifié de l'utilisateur, même si la barrière d'accès en amont est temporairement ouverte. Examinez et acceptez les conditions en amont, puis authentifiez-vous :

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Pour éviter toute requête réseau, pointez vers un snapshot que vous possédez déjà :

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Ce répertoire doit contenir `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` et `merges.txt`. Voir la section Licence ci-dessous pour ce que les conditions du checkpoint autorisent.

## Prédire

<code-tabs name="predict" />

L'API de tâches standard couvre quatre tâches, chacune associée à une cible MODUS : `depth` à la profondeur relative (`result.depth_map`), `normal` aux normales de surface (`result.normal_map`), `edge` aux contours de type Canny (`result.edges`), et `detect` aux boîtes COCO-80 (`result.boxes`) sauf si `set_classes()` la bascule en grounding textuel. `set_task()` passe de l'une à l'autre sur le même modèle chargé. La recette publiée utilise dix étapes d'échantillonnage de flux, avec une guidance texte de 4.0 et une guidance image de 2.0 ; remplacez-les avec `inference_steps=`, `inference_cfg=` et `inference_image_cfg=` à la construction.

`any2any()` donne accès à la surface d'analyse publique plus large : une à trois entrées dérivées d'une image (`rgb`, `depth`, `normal`, `canny`/`edge`), plus un texte auxiliaire optionnel, composées vers l'une quelconque de ces cibles : profondeur, normales, contours, contours dérivés de SAM, détection COCO ou grounding textuel. Toutes les entrées dérivées d'une image doivent décrire le même canevas aligné ; LibreMODUS rejette les largeurs et les hauteurs discordantes au lieu de les redimensionner indépendamment. `chain=(...)` génère des cibles intermédiaires et les réinjecte dans le même contexte, dans la limite du budget d'entraînement à trois conditions du checkpoint. `verify=N` (N >= 2) génère N candidats et garde celui qui obtient le meilleur score à un contrôle de cohérence interne contraint, exposé par `result.verification_score`.

`dtype="bf16"` (la valeur par défaut) correspond à la précision du checkpoint publié ; `dtype="fp8"` stocke les poids linéaires éligibles du tronc décodeur en E4M3 avec une échelle par canal de sortie, effectue une conversion unique vers un cache local sous `~/.cache/libreyolo/modus/fp8`, et déquantifie vers le dtype d'entrée à chaque multiplication matricielle : il joue donc sur la mémoire, pas sur l'exactitude au niveau des activations.

`train()`, `val()` et `export()` lèvent toutes une exception : LibreMODUS fonctionne en inférence seule, la validation sur dataset n'est pas proposée, et il n'existe aucun chemin d'export ONNX, TensorRT ou TFLite. Le `predict()` par batch et l'augmentation au moment du test ne sont pas non plus pris en charge ; chaque appel traite une seule image.

## Licence

<provenance-box>

LibreYOLO n'héberge ni ne met en miroir le checkpoint MODUS où que ce soit, y compris sur sa propre organisation Hugging Face : son chargement récupère toujours la révision épinglée directement depuis EPFL-VILAB/MODUS, ou lit un snapshot déjà présent sur le disque à `checkpoint_path`.

</provenance-box>

## Citation

<citation-block />
