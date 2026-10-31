---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3 : segmentation guidable et par concept dans LibreYOLO'
description: >-
  Utilisez SAM 3 dans LibreYOLO pour la segmentation par point, boîte et concept
  textuel. Installez et prédisez avec le checkpoint large, à accès restreint
  sous la SAM License de Meta.
lead: >-
  SAM 3 étend SAM en ajoutant une requête par concept textuel aux points et
  boîtes habituels. Une expression comme « yellow school bus » renvoie ainsi
  toutes les instances correspondantes. LibreYOLO prend en charge son chemin
  image par une factory LibreSAM dédiée, distincte de la factory de détecteurs
  LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - segmentation guidable
  - segmentation par concept
  - requête textuelle
  - requête par point
  - requête par boîte
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Requêtes par point et par boîte
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "sam3" est la seule taille ("large") ; alias : "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Requête par point : [x, y] en coordonnées pixel, étiquette 1 = premier
        plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un polygone par masque

        print(result.boxes.xyxy)    # boîte ajustée dérivée du masque


        # Requête par boîte au lieu d'un point.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Requête textuelle (concept)
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Recherche toutes les instances correspondant à l'expression, pas une
        seule.

        # text= est mutuellement exclusif avec points, bboxes, labels et masks.

        result = model.predict(SAMPLE_IMAGE, text="a person")

        print(result.names)         # {0: "a person"}

        print(result.boxes.conf)    # score de détection PCS par instance
    - label: 'Encoder une fois, fournir plusieurs requêtes'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # L'encodeur d'image est la partie coûteuse. set_image() l'exécute une
        fois ;

        # chaque appel predict() suivant réutilise l'embedding en cache. Un
        appel

        # text= réencode en interne, car le tracker et l'encodeur de
        segmentation

        # par concept ne partagent pas de cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Installer

SAM 3 nécessite l'extra `sam`, qui installe `transformers` et `timm`.

```bash
pip install "libreyolo[sam]"
```

L'accès aux poids est restreint : rendez-vous sur
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), acceptez
la SAM License de Meta, puis exécutez `hf auth login` (ou définissez `HF_TOKEN`)
avant le premier téléchargement. LibreYOLO journalise un avis de licence lors du
premier téléchargement de cette famille.

## Prédire

`LibreSAM(...)` (ou `LibreSAM3(...)`, propre à la famille) constitue un point
d'entrée distinct de `LibreYOLO(...)` : il renvoie un segmenteur guidable et non
un détecteur, car une passe n'a ici aucun sens sans requête. Il n'existe pas de
commande CLI `libreyolo predict` pour cette famille ; utilisez l'API Python.
Seule l'inférence sur image est prise en charge ; les modèles vidéo de SAM 3
dépassent ici le périmètre.

<code-tabs name="predict" />

Le chemin par point ou boîte correspond au reste de la famille SAM : une requête
par point accepte `[x, y]` pour un objet ou `[[x, y], ...]` pour plusieurs,
`labels` marque chaque point par `1` (premier plan) ou `0` (arrière-plan), et une
requête par boîte prend `[x1, y1, x2, y2]` ou une liste de boîtes. Sur ce chemin,
`conf` filtre selon la qualité prédite du masque (IoU), et non une confiance de
détection.

Le chemin `text=` est l'ajout de SAM 3 : une chaîne de concept renvoie chaque
instance correspondante dans l'image au moyen de Promptable Concept Segmentation
et ne peut pas être combinée avec des points, boîtes, étiquettes ou masques. Dans
ce cas, `conf` correspond au score de détection PCS plutôt qu'à l'IoU du masque.
Conserver sa valeur par défaut applique le propre seuil de 0.3 du modèle, tandis
que `conf=0.0` garde chaque candidat. Le dictionnaire `names` renvoyé associe
l'identifiant de classe `0` à la chaîne de concept demandée, puisqu'un masque
guidable ne possède autrement aucun ensemble de classes fixe. `device=` déplace
le modèle et, si une session `set_image()` est active, son embedding mis en
cache. `train()`, `val()`, `export()` et `track()` lèvent tous
`NotImplementedError` pour cette famille : SAM 3 est réservé à la prédiction
dans LibreYOLO et le suivi vidéo dépasse le périmètre. Consultez la
[prédiction](/docs/predict) pour les types de sources.

## Variantes

Une seule taille, large, avec une entrée fixe de 1008 px. SAM 3.1 n'est pas pris
en charge : son implémentation possède une licence personnalisée qui ne peut pas
être intégrée à ce dépôt MIT, et la version de Transformers dont dépend
LibreYOLO ne charge pas encore son format de checkpoint.

## Licence

<provenance-box>

LibreYOLO n'héberge aucune copie des poids SAM 3 et ne les redistribue pas.
`LibreSAM("sam3")` les télécharge directement depuis le dépôt à accès restreint
`facebook/sam3` de Meta sur Hugging Face, qui exige l'acceptation de la SAM
License de Meta et une authentification avant le premier téléchargement.

</provenance-box>

## Citation

<citation-block />
