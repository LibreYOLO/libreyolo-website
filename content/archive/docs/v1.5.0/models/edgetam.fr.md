---
title: EdgeTAM
families:
  - edgetam
seo_title: "EdgeTAM\_: segmentation guidée par prompts sur l'appareil dans LibreYOLO"
description: >-
  Utilisez EdgeTAM dans LibreYOLO pour la segmentation guidée par prompts de
  point et de boîte, conçue pour la vitesse sur l'appareil. Installez et
  prédisez le checkpoint sous Apache-2.0.
lead: >-
  EdgeTAM est une variante de SAM 2 pensée pour tourner sur l'appareil,
  construite pour la vitesse d'inférence sur mobile tout en conservant le même
  flux de travail guidé par prompts de point et de boîte. LibreYOLO prend en
  charge son chemin de segmentation d'images via une factory LibreSAM dédiée,
  distincte de la factory de détecteurs LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - segmentation guidée par prompts
  - segmentation interactive python
  - segmenter un objet en un clic
  - segmentation sur l'appareil
  - prompt de point
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompts de point et de boîte
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAM a une seule taille, "edge". Alias: "edgetam", "edge-tam",
        # "edgetam-edge".
        model = LibreSAM("edgetam")

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
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # L'encodeur d'image est la partie coûteuse. set_image() le lance une
        # fois; chaque predict() ensuite réutilise l'embedding mis en cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Installation

EdgeTAM demande l'extra `sam`, qui installe `transformers` et `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prédire

`LibreSAM(...)` (ou le `LibreEdgeTAM(...)` propre à la famille) est un point
d'entrée distinct de `LibreYOLO(...)` : il renvoie un segmenteur guidé par
prompts plutôt qu'un détecteur, parce qu'ici une passe avant n'a aucun sens
sans prompt spatial. Il n'existe pas de commande CLI `libreyolo predict` pour
cette famille ; utilisez l'API Python. Seule la segmentation d'images est prise
en charge ; le suivi vidéo d'EdgeTAM sort du cadre ici.

<code-tabs name="predict" />

Un prompt de point accepte `[x, y]` pour un objet, `[[x, y], ...]` pour
plusieurs, ou des tableaux numpy ; `labels` marque chaque point `1` (premier
plan) ou `0` (arrière-plan) et vaut par défaut tout en premier plan. Un prompt
de boîte prend `[x1, y1, x2, y2]` ou une liste de boîtes, un masque par boîte.
Omettre les deux prompts segmente l'image entière en soumettant une grille
dense de prompts et en gardant les masques confiants qui ne se recouvrent pas ;
ce mode « tout segmenter » est simplifié par rapport au générateur automatique
de masques de référence et peut sous-segmenter les scènes chargées, si bien
qu'un vrai prompt de point ou de boîte reste le chemin précis. `conf` filtre
sur la qualité de masque prédite (IoU), pas sur une confiance de détection :
passez `0.0` pour conserver tous les candidats. `multimask=True` renvoie les
trois masques d'ambiguïté tout-contre-partie de SAM pour chaque prompt au lieu
du seul meilleur. `device=` déplace le modèle et, si une session `set_image()`
est active, son embedding mis en cache. Chaque masque porte l'id de classe `0`,
nommé `"object"`, puisqu'un masque guidé par prompts n'a pas d'ensemble de
classes fixe. `train()`, `val()`, `export()` et `track()` lèvent tous
`NotImplementedError` pour cette famille : l'inférence sur images est ce que
LibreYOLO prend en charge ici. Voir [la prédiction](/docs/predict) pour les
types de sources.

## Variantes

Une seule taille, edge, à une résolution d'entrée fixe, si bien que choisir
cette famille plutôt que le reste du niveau SAM est une décision matérielle
plus qu'une question de taille : EdgeTAM existe spécifiquement pour l'inférence
sur l'appareil, sous contraintes de ressources.

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
