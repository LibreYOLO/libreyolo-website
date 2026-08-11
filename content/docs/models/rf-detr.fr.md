---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR : entraîner, affiner et exporter sous MIT'
description: >-
  Utilisez RF-DETR dans LibreYOLO pour la détection, la segmentation
  d'instances, la pose et les boîtes orientées. Installez, prédisez, entraînez,
  validez et exportez, entièrement sous licence MIT.
lead: >-
  Un detection transformer qui prédit un ensemble fixe d'objets au lieu d'une
  grille dense, ce qui élimine le besoin de NMS lors de l'inférence. LibreYOLO
  le prend en charge pour quatre tâches.
keywords:
  - RF-DETR
  - detection transformer temps réel
  - DETR
  - détection d'objets
  - segmentation d'instances
  - estimation de pose
  - boîtes orientées
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, détection sur vidéo à 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Vidéo
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Toute source acceptée par la bibliothèque : fichier, dossier, URL,
        # index de webcam, flux RTSP ou liste .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() renvoie un simple dict, pas un objet
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Comparer à COCO
      language: bash
      code: |
        # Le YAML COCO inclus contient un script de téléchargement intégré ; une
        # autorisation explicite est requise sauf si le dataset est déjà local.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Arguments acceptés pour tous les formats :

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" est un alias de tensorrt, "litert" de tflite.

        #   imgsz     int ou (hauteur, largeur). Utilise par défaut la
        résolution

        #             native du checkpoint.

        #   batch     int, 1 par défaut.

        #   half      bool, export en FP16. False par défaut.

        #   int8      bool, export en INT8. False par défaut. Nécessite `data`.

        #   data      chemin d'un YAML de dataset pour calibrer int8.

        #   fraction  float, part du dataset de calibration. 1.0 par défaut.

        #   dynamic   bool, axes dynamiques. True par défaut.

        #   simplify  bool, simplification du graphe ONNX. True par défaut.

        #   opset     int, opset ONNX. Choisi par famille s'il n'est pas fourni.

        #   device    str, appareil de traçage. Utilise celui du modèle par
        défaut.

        #   output_path  str, nom dérivé du checkpoint par défaut.

        #   verbose   bool, False par défaut.

        #   allow_download_scripts  bool, False par défaut. Autorise le code

        #             Python d'un YAML de dataset qui doit être téléchargé.

        #

        # Quelques formats acceptent leurs propres arguments, comme une
        plateforme

        # cible RKNN. Ils sont documentés sur la page de chaque format.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La factory utilise le suffixe du fichier : un artefact exporté se
        charge

        # comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreRFDETRs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
    - label: Sans LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # L'exécution directe du graphe impose votre propre pré/post-traitement.

        # Inspectez la signature avant de connecter quoi que ce soit.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Installer

RF-DETR nécessite son propre extra, qui installe `transformers` pour le backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est le même pour toutes les familles, remplacer le
détecteur ne demande donc de modifier qu'une ligne. `conf` et `max_det` filtrent
la sélection des requêtes ; aucune étape NMS n'est à régler. Consultez la
[prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Variantes

Quatre tailles et quatre tâches partagent une même architecture : la
segmentation, la pose et les boîtes orientées réutilisent le décodeur de
détection avec une tête différente, et acceptent donc les mêmes arguments. Les
tailles comportent un nombre de paramètres similaire et diffèrent principalement
par la résolution d'entrée.

<benchmark-table task="detect" />

<va-embed />

## Entraîner

Pour les quatre tâches, l'entraînement part d'un checkpoint publié. RF-DETR
répertorie `pretrained` parmi les arguments ignorés par son entraîneur natif.
Fournir `pretrained=False` ne produit donc pas ici un modèle initialisé
aléatoirement.

<code-tabs name="train" />

Deux arguments comptent davantage ici que sur un détecteur CNN. Conservez `lr0`
à `1e-4` ou moins, car les detection transformers divergent avec des learning
rates qu'un modèle YOLO tolère. Laissez `imgsz` à la résolution native du
checkpoint sauf si vous avez une raison de la modifier. L'entrée doit être
divisible sans reste par le produit de la taille de patch du backbone et du
nombre de fenêtres. LibreYOLO le vérifie avant le début de l'exécution et indique
les tailles valides les plus proches.

Consultez l'[entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurés sur tout dataset dans le format
utilisé pour l'entraînement.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge par `LibreYOLO()` à partir de son suffixe de
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint et
renvoie le même objet `Results`. L'exécution du graphe dans un runtime brut, sans
installation de LibreYOLO, est également prise en charge, mais vous devez alors
écrire vous-même le prétraitement et le post-traitement.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
