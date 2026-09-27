---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2 : segmentation d''images guidable dans LibreYOLO'
description: >-
  Utilisez SAM 2 dans LibreYOLO pour la segmentation guidable par point et par
  boîte. Installez et prédisez avec les checkpoints tiny, small, base-plus et
  large sous Apache-2.0.
lead: >-
  SAM 2 étend SAM avec une architecture à mémoire de streaming conçue pour la
  vidéo, et transforme un clic sur un point ou une boîte en masque d'objet.
  LibreYOLO prend en charge son chemin de segmentation d'images par une factory
  LibreSAM dédiée, distincte de la factory de détecteurs LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - segmentation guidable
  - segmentation interactive
  - requête par point
  - requête par boîte
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Requêtes par point et par boîte
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Alias de tailles : "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (et formes courtes "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


        # Requête par point : [x, y] en coordonnées pixel, étiquette 1 = premier
        plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un polygone par masque

        print(result.boxes.xyxy)    # boîte ajustée dérivée du masque


        # Requête par boîte au lieu d'un point.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sans aucune requête, l'image entière est segmentée (générateur
        automatique

        # de masques simplifié, pas la version de référence exhaustive).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Encoder une fois, fournir plusieurs requêtes'
      language: python
      code: >
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # La classe propre à la famille prend la taille sans le préfixe "sam2-".

        model = LibreSAM2("large")


        # L'encodeur d'image est la partie coûteuse. set_image() l'exécute une
        fois ;

        # chaque appel predict() suivant réutilise l'embedding mis en cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Installer

SAM 2 nécessite l'extra `sam`, qui installe `transformers` et `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prédire

`LibreSAM(...)` (ou `LibreSAM2(...)`, propre à la famille) constitue un point
d'entrée distinct de `LibreYOLO(...)` : il renvoie un segmenteur guidable et non
un détecteur, car une passe n'a ici aucun sens sans requête spatiale. Il n'existe
pas de commande CLI `libreyolo predict` pour cette famille ; utilisez l'API
Python. Seule la segmentation d'images est prise en charge ; le suivi avec
mémoire vidéo de SAM 2 dépasse ici le périmètre.

<code-tabs name="predict" />

Une requête par point accepte `[x, y]` pour un objet, `[[x, y], ...]` pour
plusieurs ou des tableaux numpy. `labels` marque chaque point par `1` (premier
plan) ou `0` (arrière-plan) et utilise par défaut le premier plan pour tous. Une
requête par boîte prend `[x1, y1, x2, y2]` ou une liste de boîtes, avec un masque
par boîte. Si vous omettez les deux types de requêtes, l'image entière est
segmentée en guidant le modèle avec une grille dense et en conservant les masques
confiants qui ne se chevauchent pas. Ce mode « tout segmenter » est simplifié par
rapport au générateur automatique de masques de référence et peut sous-segmenter
les scènes très chargées. Une véritable requête par point ou boîte constitue
donc le chemin précis. `conf` filtre selon la qualité prédite du masque (IoU), et
non une confiance de détection : fournissez `0.0` pour conserver tous les
candidats. `multimask=True` renvoie les trois masques d'ambiguïté entre objet
entier et partie de SAM pour chaque requête, au lieu du seul meilleur.
`device=` déplace le modèle et, si une session `set_image()` est active, son
embedding mis en cache. Chaque masque porte l'identifiant de classe `0`, nommé
`"object"`, puisqu'un masque guidable ne possède aucun ensemble de classes fixe.
`train()`, `val()`, `export()` et `track()` lèvent tous `NotImplementedError`
pour cette famille : LibreYOLO ne prend ici en charge que l'inférence sur image.
Consultez la [prédiction](/docs/predict) pour les types de sources.

## Variantes

Quatre tailles de backbone Hiera sont proposées, tiny, small, base-plus et large,
toutes à la même résolution d'entrée. Aucun benchmark d'exactitude ou de latence
n'est publié pour cette famille. Le choix d'une taille échange donc directement
le poids de l'encodeur contre la qualité du masque : tiny est la plus rapide à
encoder, large la plus lourde.

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
