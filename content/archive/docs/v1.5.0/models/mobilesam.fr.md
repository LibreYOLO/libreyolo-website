---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM : segmentation légère guidée par prompts dans LibreYOLO'
description: >-
  Utilisez MobileSAM dans LibreYOLO pour la segmentation guidée par prompts de
  point et de boîte avec un encodeur TinyViT. Installez et prédisez le
  checkpoint tiny sous Apache-2.0.
lead: >-
  MobileSAM remplace l'encodeur d'image ViT-H de SAM par un encodeur TinyViT
  distillé, si bien que le même flux de travail guidé par prompts de point et de
  boîte tourne sur du matériel plus léger. LibreYOLO en embarque un portage
  natif via une factory LibreSAM dédiée, distincte de la factory de détecteurs
  LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - segmentation guidée par prompts
  - segmentation interactive python
  - segmenter un objet en un clic
  - prompt de point
  - segmentation légère
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de point et de boîte
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # MobileSAM a une seule taille, "tiny": aucun autre alias n'est utile.
        model = LibreSAM("mobilesam")

        # Un prompt de point: [x, y] en pixels, label 1 = premier plan.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # un polygone par masque
        print(result.boxes.xyxy)    # boîte ajustée dérivée du masque

        # Un prompt de boîte au lieu d'un point.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # Sans aucun prompt, l'image entière est segmentée (un générateur
        # automatique de masques simplifié, pas celui de référence).
        result = model.predict(SAMPLE_IMAGE)
    - label: 'Encoder une fois, envoyer plusieurs prompts'
      language: python
      code: |
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE

        model = LibreMobileSAM()

        # L'encodeur d'image est la partie coûteuse. set_image() le lance une
        # fois; chaque predict() ensuite réutilise l'embedding mis en cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Installation

MobileSAM demande l'extra `sam` : le téléchargement des poids propre à
LibreYOLO passe toujours par l'outillage de snapshot Hugging Face de
`transformers`, même si l'inférence tourne sur un décodeur natif, hors
`transformers`.

```bash
pip install "libreyolo[sam]"
```

## Prédire

`LibreSAM(...)` (ou le `LibreMobileSAM(...)` propre à la famille) est un point
d'entrée distinct de `LibreYOLO(...)` : il renvoie un segmenteur guidé par
prompts plutôt qu'un détecteur, parce qu'ici une passe avant n'a aucun sens
sans prompt spatial. Il n'existe pas de commande CLI `libreyolo predict` pour
cette famille ; utilisez l'API Python.

<code-tabs name="predict" />

Un prompt de point accepte `[x, y]` pour un objet, `[[x, y], ...]` pour
plusieurs, ou des tableaux numpy ; `labels` marque chaque point `1` (premier
plan) ou `0` (arrière-plan) et vaut par défaut tout en premier plan. Un prompt
de boîte prend `[x1, y1, x2, y2]` ou une liste de boîtes, un masque par boîte.
Omettre les deux prompts segmente l'image entière en soumettant une grille
dense de prompts et en gardant les masques confiants qui ne se recouvrent pas ;
ce mode « tout segmenter » est simplifié par rapport au générateur automatique
de masques de référence et peut sous-segmenter les scènes chargées, si bien
qu'un vrai prompt de point ou de boîte reste le chemin précis. `conf` filtre
sur la qualité de masque prédite (IoU), pas sur une confiance de détection :
passez `0.0` pour conserver tous les candidats. `multimask=True` renvoie les
trois masques d'ambiguïté tout-contre-partie de SAM pour chaque prompt au lieu
du seul meilleur. `device=` déplace le modèle et, si une session `set_image()`
est active, son embedding mis en cache. Chaque masque porte l'id de classe `0`,
nommé `"object"`, puisqu'un masque guidé par prompts n'a pas d'ensemble de
classes fixe. `train()`, `val()`, `export()` et `track()` lèvent tous
`NotImplementedError` pour cette famille : MobileSAM est en prédiction seule
dans LibreYOLO. Voir [la prédiction](/docs/predict) pour les types de sources.

## Variantes

Une seule taille, tiny, à une entrée fixe de 1024 px : MobileSAM livre un
unique encodeur TinyViT plutôt que l'échelle base/large/huge proposée par
SAM-1.

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
