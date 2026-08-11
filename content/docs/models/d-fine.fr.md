---
title: D-FINE
families:
  - dfine
seo_title: "D-FINE\_: faire du fine-tuning, valider et exporter sous licence MIT"
description: >-
  Utilisez D-FINE dans LibreYOLO pour la détection d'objets et la segmentation
  d'instances. Installez, prédisez, faites du fine-tuning, validez et exportez,
  avec du code sous licence MIT.
lead: >-
  Un transformer de détection qui reformule la régression des boîtes comme une
  distribution de probabilité sur chaque bord de boîte, affinée au fil des
  couches du décodeur. LibreYOLO le prend en charge pour la détection et la
  segmentation d'instances.
keywords:
  - D-FINE
  - detection transformer
  - détection d'objets temps réel
  - segmentation d'instances python
  - DETR
  - fine-tuning D-FINE
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le suffixe -seg du nom de fichier sélectionne la tête de masques,
        # aucun argument task n'est donc nécessaire ici.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentation d'instances
      language: bash
      code: >
        # Reprend depuis les poids de segmentation publiés, tête de masques
        incluse.

        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentation à partir de poids de détection
      language: bash
      code: |
        # Les poids de détection n'ont pas de tête de masques, c'est donc un
        # transfert explicite. La tête démarre non entraînée et n'est utile
        # qu'une fois entraînée. C'est task=segment qui autorise le transfert.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentation d'instances
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # masques
        print(metrics["metrics/mAP50-95(B)"])   # boîtes
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factory route selon l'extension du fichier, donc un artefact
        exporté

        # se charge comme un checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreDFINEn.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Installation

D-FINE ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

Le fine-tuning par adaptateurs avec `lora=True` fait exception, et demande
l'extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoie chaque famille, si bien que
remplacer le détecteur par un autre tient en une seule ligne. Un nom de fichier
en `-seg` résout de lui-même vers la tâche de segmentation, et `result.masks`
porte alors les masques d'instances à côté des boîtes. `conf` et `max_det`
filtrent la sélection des queries ; `iou` est accepté par parité d'API mais n'a
aucun effet, car le décodeur est un set predictor sans étape de NMS. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Cinq tailles. Elles tournent toutes à la même résolution d'entrée, la table les
sépare donc par nombre de paramètres et par exactitude.

<benchmark-table task="detect" />

<va-embed />

La segmentation réutilise le backbone, l'encodeur et le décodeur de détection et
y ajoute une tête de masques, si bien qu'un checkpoint `-seg` prend les mêmes
arguments que son homologue de détection. La famille RT-DETRv4 de LibreYOLO est
écrite comme une sous-classe du wrapper D-FINE : elle hérite de cette lignée de
décodeur, puis restreint sa liste de tâches à la seule détection, car elle ne
porte pas de tête de masques.

## Entraîner

L'entraînement démarre depuis un checkpoint publié, pour les deux tâches.

<code-tabs name="train" />

Laissé à ses réglages par défaut, l'entraîneur fait 132 époques à `lr0=2e-4`
avec `amp=False`, un batch de 16 et un early stopping après 50 époques sans
amélioration. Les poids de détection sont un point de départ légitime pour un
entraînement de segmentation, mais seulement comme transfert explicite, puisque
la tête de masques démarre non entraînée et renverrait sinon des masques sans
signification. C'est le passage de `task=segment` à la CLI qui l'autorise. La
voie Python est plus étroite : `LibreDFINE` doit être construit directement avec
`allow_detect_to_segment_transfer=True`, car la factory `LibreYOLO()` n'accepte
pas cet argument, et la construction directe ne télécharge rien, si bien que le
fichier de poids doit déjà être présent sur le disque.

`lora=True` s'applique à la détection. L'entraînement en segmentation le refuse
et renvoie plutôt vers `freeze='backbone'`, car la tête de masques n'a pas été
testée avec des adaptateurs. Sur Apple silicon, l'entraîneur bascule tout le run
sur CPU : la passe arrière du matmul par bins de l'Integral déclenche un échec de
compilation Metal. L'inférence sur MPS n'est pas concernée.

Voir [l'entraînement](/docs/train) pour les datasets, l'augmentation, le
multi-GPU et les loggers.

## Valider

`val()` renvoie un dictionnaire indexé par nom de métrique, et affiche les
résultats par classe tant que `verbose` reste actif.

<code-tabs name="val" />

Sur un checkpoint `-seg`, la clé `metrics/mAP50-95` sans suffixe porte le score
des masques, et le même run reporte aussi les boîtes sous `(B)` et les masques
sous `(M)`, si bien que les deux sont disponibles en une seule passe.

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. Les voies OpenVINO, Paddle, MNN et
Core AI exportent à taille de canevas fixe plutôt qu'en formes dynamiques.
[L'export](/docs/export) liste les arguments que chaque format accepte, ainsi
que les extras que quelques-uns ajoutent.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Les poids de segmentation ont une seconde source amont : leur décodeur de
masques, leur appariement de masques et leur loss de masques viennent de
ArgoHA/D-FINE-seg, également en Apache-2.0, dont le mainteneur a approuvé la
réutilisation avec attribution.

</provenance-box>

## Citation

<citation-block />
