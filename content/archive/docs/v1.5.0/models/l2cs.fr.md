---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net : l''estimation du regard dans LibreYOLO'
description: >-
  Utilisez L2CS-Net dans LibreYOLO pour l'estimation du regard en deux étapes,
  en pitch et en yaw. Installez, prédisez et exportez ; le checkpoint Gaze360
  est réservé à la recherche.
lead: >-
  L2CS-Net est un estimateur de regard en deux étapes : un détecteur de visages
  localise les visages, puis un tronc ResNet doté de deux têtes de
  classification par bins d'angle prédit le pitch et le yaw de chaque visage.
  LibreYOLO l'encapsule en inférence seule.
keywords:
  - L2CS-Net
  - estimation du regard
  - eye tracking python
  - direction du regard pitch yaw
  - Gaze360
  - détection de visages python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Aucun face_detector fourni : repli sur le détecteur de visages
        # livré avec OpenCV (Haar sur OpenCV 4, YuNet sur OpenCV 5), donc
        # aucun téléchargement au-delà du checkpoint L2CS lui-même.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Source des visages
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Passez à L2CS les boîtes d'un détecteur déjà exécuté.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Ou nommez un détecteur de visages livré avec la bibliothèque.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Utiliser le fichier exporté
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Le graphe exporté se limite au tronc ResNet et aux deux têtes de
        # bins d'angle : il prend un crop de visage 448x448 prétraité et
        # renvoie des (yaw_logits, pitch_logits) bruts, pas des angles
        # décodés. Softmax, espérance des bins et conversion en degrés
        # restent en Python ; voir
        # libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Installation

L2CS-Net ne demande aucun extra pour construire, prédire ou exporter un modèle
dont vous avez déjà le checkpoint.

```bash
pip install libreyolo
```

Le seul checkpoint que LibreYOLO peut récupérer automatiquement, un ResNet-50
entraîné sur Gaze360, se télécharge via `gdown` plutôt que depuis un simple
miroir HTTP, car il réside sur le Google Drive de l'auteur et non dans
l'organisation LibreYOLO. Ce chemin exige l'extra `gaze` :

```bash
pip install "libreyolo[gaze]"
```

Sans lui, LibreYOLO affiche les instructions de téléchargement manuel au lieu
d'échouer silencieusement.

## Prédire

<code-tabs name="predict" />

L2CS-Net est un estimateur en deux étapes : un détecteur de visages s'exécute
d'abord, et la tête de regard lit le pitch et le yaw dans chaque crop de visage
qu'il renvoie. Laissée à elle-même, la prédiction retombe sur le détecteur livré
avec OpenCV, si bien qu'un appel nu fonctionne sans téléchargement
supplémentaire dès que vous disposez du checkpoint L2CS lui-même. `face_boxes`
accepte les boîtes d'un détecteur que vous avez déjà exécuté ; `face_detector`
accepte `"auto"`, `"haar"`, `"yunet"`, un modèle de détection LibreYOLO ou un
simple callable. `result.gaze` porte le pitch et le yaw en radians, alignés
ligne à ligne avec `result.boxes`, les boîtes de visages détectées. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Cinq profondeurs de backbone partagent une même résolution d'entrée et prennent
les mêmes arguments. Gaze360, le dataset derrière le seul checkpoint publié, a
servi à entraîner un ResNet-50 ; les quatre autres profondeurs sont prises en
charge sur le plan architectural, mais n'ont aucun poids publié à charger.

## Exporter

<export-matrix />

<code-tabs name="export" />

## Licence

<provenance-box>

LibreYOLO n'héberge ni ne met en miroir aucun checkpoint L2CS : rien de cette
famille n'existe dans l'organisation Hugging Face LibreYOLO, contrairement à la
plupart des autres familles de ce site. Le seul checkpoint que la bibliothèque
peut récupérer automatiquement provient directement de la distribution Google
Drive de l'auteur, derrière l'avis de licence Gaze360 affiché avant le début du
transfert, et n'est pas la copie « republiée sur huggingface.co/LibreYOLO » que
laisse entendre le résumé ci-dessus.

</provenance-box>
