---
title: OpenVINO
seo_title: Exporter vers OpenVINO IR depuis LibreYOLO
description: "Convertissez un modèle LibreYOLO en OpenVINO IR\_: la paire model.xml et model.bin, la compression des poids en FP16, l'INT8 via NNCF, et l'inférence sur CPU, GPU ou NPU."
lead: >-
  OpenVINO IR est le format de runtime d'Intel, un graphe model.xml à côté d'un
  blob de poids model.bin. LibreYOLO exporte un ONNX intermédiaire, le convertit
  avec ov.convert_model, et écrit un metadata.yaml dans le même répertoire.
keywords:
  - exporter yolo openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - quantification int8 nncf
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="openvino")
    mono: true
  - label: Écrit
    value: 'Un répertoire contenant model.xml, model.bin et metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Formes
    value: "Suit l'ONNX intermédiaire\_: batch dynamique quand dynamic=True"
  - label: Précision
    value: >-
      FP32, compression des poids en FP16 (half=True), INT8 via NNCF (int8=True
      avec data=)
verification: >-
  Lu depuis libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py et pyproject.toml
  sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: >
        # L'IR est converti depuis un ONNX intermédiaire : les deux extras sont
        requis.

        pip install "libreyolo[onnx,openvino]"
    - label: INT8 nécessite en plus NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit le répertoire weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Arguments
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True garde un axe de batch dynamique dans l'IR
            half=False,       # True stocke des poids FP16
            int8=False,       # True lance la quantification post-training NNCF
            data=None,        # requis quand int8=True
            output_path=None, # None écrit weights/<stem>_openvino
        )
  int8:
    - label: INT8 avec des données de calibration
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # requis : aucun défaut pour ce format
            fraction=1.0,
        )
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Choisir le device
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" et "cpu" pointent vers CPU, "gpu" et "cuda" vers GPU,
        # tout le reste est transmis en majuscules, par exemple "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO brut
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Noms de classe, tâche et taille d'entrée sont dans metadata.yaml.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Le prétraitement et le postprocessing sont à votre charge ici.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Installation

<code-tabs name="install" />

La conversion passe par un ONNX intermédiaire, donc l'extra `onnx` fait partie du
prérequis plutôt que d'un compagnon optionnel. NNCF s'installe à part et n'est
nécessaire que pour `int8=True`.

## Export

<code-tabs name="export" />

L'artefact est un répertoire, pas un fichier. `weights/LibreYOLO9t_openvino` contient
`model.xml`, `model.bin` et `metadata.yaml`, et `_fp16` est inséré avant le suffixe
quand `half=True`. Déplacez ou copiez le répertoire entier ; les trois fichiers ne font
qu'un seul artefact.

`half=True` active `compress_to_fp16` à l'enregistrement. C'est une compression des poids
dans l'IR, pas un changement de la précision d'inférence que le device choisit à
l'exécution.

### INT8

<code-tabs name="int8" />

`int8=True` lance la quantification post-training de NNCF sur un loader de calibration
LibreYOLO avec le preset mixed, et `data` est obligatoire : ce format n'a pas de repli
sur huit images. Un NNCF absent lève une `ImportError` qui nomme la commande
d'installation.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` reconnaît tout répertoire contenant `model.xml` et renvoie le même objet
`Results` que le checkpoint, en lisant les noms de classe, la tâche, la taille d'entrée
et le schéma de pose depuis `metadata.yaml`.

La chaîne de device est remappée plutôt que transmise telle quelle. `auto` et `cpu`
compilent tous deux pour le CPU, `gpu` et `cuda` compilent tous deux pour le GPU, et
toute autre valeur est passée en majuscules à OpenVINO, ce qui est la façon d'atteindre
une cible NPU.

Le troisième snippet s'adresse aux lecteurs qui n'ont pas LibreYOLO installé. Le
prétraitement, le décodage, le NMS et le rééchelonnement des coordonnées y sont à votre
charge, et les noms de classe n'existent que dans `metadata.yaml`.

## Contraintes

Un IR privé de son `metadata.yaml` se charge quand même, mais le backend se rabat alors
sur 80 classes et la tâche de détection, ce qui est faux pour tout le reste. Gardez le
répertoire intact.

Bloqués avant le tracing : la segmentation YOLO9, la segmentation RTMDet-Ins, la
détection SSD, Faster R-CNN et RetinaNet, et le matting BiRefNet ou FeyNobg, où
OpenVINO 2026.2 n'arrive pas à abaisser l'opération ONNX standard `DeformConv-19` du
décodeur de matte partagé.

Quand une combinaison n'est ni validée ni bloquée, le chemin du convertisseur est
disponible et le projet n'a pas enregistré de parité du runtime OpenVINO pour elle.
Plusieurs combinaisons sont validées avec un contexte explicite attaché, par exemple la
segmentation sémantique DeepLabV3 à une entrée fixe de 520 par 520 sur OpenVINO 2026.2
avec la précision d'inférence par défaut du CPU, et l'estimation du regard L2CS sur un
crop de visage fixe de 448 par 448. `libreyolo formats` affiche ce contexte pour chaque
combinaison.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
