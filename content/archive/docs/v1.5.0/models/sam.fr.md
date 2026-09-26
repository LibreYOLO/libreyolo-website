---
title: SAM
families:
  - sam
seo_title: "SAM (Segment Anything)\_: prédire des masques dans LibreYOLO"
description: >-
  Utilisez SAM dans LibreYOLO pour la segmentation guidable par points et
  bounding boxes. Installez-le et lancez des prédictions avec les checkpoints
  base, large et huge sous licence Apache-2.0.
lead: >-
  SAM (Segment Anything) transforme un clic sur un point ou une bounding box en
  masque d'objet. LibreYOLO le charge au moyen d'une fabrique LibreSAM dédiée,
  distincte de la fabrique de détecteurs LibreYOLO(), car un modèle guidable
  nécessite une signature d'appel différente.
keywords:
  - SAM
  - Segment Anything
  - segmentation guidable
  - segmentation interactive
  - prompt point
  - prompt bounding box
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts par point et bounding box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" télécharge automatiquement facebook/sam-vit-base à la première
        utilisation.

        # Autres tailles : "large", "huge" (ainsi que "b"/"l"/"h").

        model = LibreSAM("base")


        # Un prompt par point : [x, y] en pixels, étiquette 1 = premier plan.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # un polygone par masque

        print(result.boxes.xyxy)    # bounding box ajustée dérivée du masque


        # Un prompt par bounding box à la place d'un point.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Sans prompt, l'image entière est segmentée (générateur de masques

        # automatique simplifié, différent de la version de référence
        exhaustive).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Encoder une fois, guider plusieurs fois'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # L'encodeur d'image est la partie coûteuse. set_image() ne l'exécute
        qu'une fois ;

        # chaque appel predict() suivant réutilise l'embedding mis en cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Installer

SAM nécessite l'extra `sam`, qui installe `transformers` et `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prédire

`LibreSAM(...)` est un point d'entrée distinct de `LibreYOLO(...)`\u00a0: il
renvoie un segmenteur guidable plutôt qu'un détecteur, car une passe forward
n'a ici aucun sens sans prompt spatial. Il n'existe aucune commande CLI
`libreyolo predict` pour cette famille\u00a0; utilisez l'API Python.

<code-tabs name="predict" />

Un prompt par point accepte `[x, y]` pour un objet, `[[x, y], ...]` pour
plusieurs objets ou des tableaux numpy\u00a0; `labels` associe à chaque point la
valeur `1` (premier plan) ou `0` (arrière-plan), tous les points étant au
premier plan par défaut. Un prompt par bounding box prend
`[x1, y1, x2, y2]` ou une liste de bounding boxes, avec un masque par bounding
box. Si vous omettez les deux prompts, l'image entière est segmentée en
appliquant des prompts sur une grille dense et en conservant les masques
fiables qui ne se chevauchent pas\u00a0; ce mode «\u00a0tout segmenter\u00a0» est simplifié
par rapport au générateur automatique de masques de référence et peut
sous-segmenter les scènes encombrées. Un véritable prompt par point ou
bounding box est donc la méthode précise. `conf` filtre selon la qualité de
masque prédite (IoU), et non selon une confiance de détection\u00a0: transmettez
`0.0` pour conserver chaque candidat. `multimask=True` renvoie, pour chaque
prompt, les trois masques d'ambiguïté objet complet ou partie de SAM plutôt
que le seul meilleur masque. `device=` déplace le modèle et, si une session
`set_image()` est active, son embedding mis en cache. Chaque masque porte
l'identifiant de classe `0`, nommé `"object"`, puisqu'un masque guidable ne
dispose d'aucun ensemble de classes fixe. `train()`, `val()`, `export()` et
`track()` lèvent tous `NotImplementedError` pour cette famille\u00a0: dans LibreYOLO,
SAM sert uniquement à la prédiction et le suivi vidéo est hors périmètre.
Consultez la [prédiction](/docs/predict) pour les types de sources.

## Variantes

Trois tailles d'encodeur d'image ViT\u00a0: base, large et huge, toutes avec une
entrée fixe de 1024\u00a0px. Aucun benchmark d'exactitude ou de latence n'est encore
publié pour cette famille. Le choix d'une taille met donc directement en
balance le poids de l'encodeur et la qualité du masque\u00a0: base est la plus
rapide à encoder, huge la plus lourde.

## Licence

<provenance-box>

LibreYOLO n'héberge pas sa propre copie des poids de SAM-1. `LibreSAM("base")`,
`"large"` et `"huge"` les téléchargent directement depuis les dépôts
`facebook/sam-vit-base`, `facebook/sam-vit-large` et `facebook/sam-vit-huge`
de Meta sur Hugging Face, chacun y étant étiqueté Apache-2.0 indépendamment de
LibreYOLO.

</provenance-box>

## Citation

<citation-block />

