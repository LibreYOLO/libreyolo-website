---
title: Core ML
seo_title: Exporter vers Core ML depuis LibreYOLO
description: "Exportez un détecteur LibreYOLO vers un .mlpackage Core ML\_: le contrat d'entrée ImageType, le FP16, les compute units, le NMS embarqué et les quatre familles prises en charge."
lead: >-
  Core ML est le format de modèles on-device d'Apple. LibreYOLO trace le
  détecteur derrière un wrapper de prétraitement propre à chaque famille, de
  sorte que le graphe converti reçoit toujours une entrée image RGB canonique,
  puis écrit un .mlpackage au format ML Program avec les métadonnées du modèle
  attachées.
keywords:
  - exporter yolo coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - nms embarqué coreml
  - yolo sur ios
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreml")
    mono: true
  - label: Écrit
    value: Un bundle .mlpackage (un répertoire) au format ML Program
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") sur macOS
    mono: true
  - label: Formes
    value: Fixes. L'entrée est un ct.ImageType à forme rigide.
  - label: Précision
    value: 'FP32, FP16 (half=True). Pas d''INT8.'
  - label: Familles
    value: 'Détection uniquement, pour yolox, yolo9, rtdetr et rfdetr'
verification: >-
  Lu depuis libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py et pyproject.toml
  sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit le bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Arguments
      language: python
      code: >
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True convertit en précision de calcul FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None écrit weights/<stem>.mlpackage
        )


        # dynamic est accepté, mais l'entrée est un ct.ImageType de forme fixe,

        # et les métadonnées embarquées enregistrent dynamic=False dans tous les
        cas.
  nms:
    - label: Intégrer la couche NMS d'Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Détection YOLOX et YOLO9 uniquement, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Via LibreYOLO, sur macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # ou cpu_and_ne pour fixer le Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools brut
      language: python
      code: >
        import coremltools as ct

        from PIL import Image


        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")

        print(mlmodel.user_defined_metadata["model_family"])

        print(mlmodel.user_defined_metadata["names"])


        # L'entrée est une image nommée "image" à la taille fixe d'export.

        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))

        out = mlmodel.predict({"image": image})

        print({name: value.shape for name, value in out.items()})


        # Le letterboxing et le postprocessing sont à votre charge sur ce
        chemin.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Installation

<code-tabs name="install" />

La prédiction exige macOS. `LibreYOLO()` refuse un `.mlpackage` sur toute autre
plateforme avec un message qui nomme la plateforme courante, et la matrice de support
enregistre ces combinaisons comme disponibles au motif que la parité au runtime exige un runner macOS.

## Export

<code-tabs name="export" />

Le bundle est écrit dans `weights/` sous le stem du checkpoint, avec `_fp16`
ajouté quand `half=True`. Un `.mlpackage` est un répertoire, donc copiez l'arbre entier.

Chaque famille est tracée derrière un wrapper de prétraitement, de sorte que le graphe
converti reçoit une seule entrée canonique : RGB, `scale=1/255`, sans biais, déclarée comme
`ct.ImageType`. Le wrapper absorbe la convention propre à chaque famille, à savoir BGR dans
la plage 0 à 255 pour YOLOX, moyenne et écart type ImageNet pour RF-DETR,
et identité pour YOLO9 et RT-DETR. C'est pourquoi un consommateur Core ML alimente une
image ordinaire plutôt qu'un tenseur spécifique à la famille.

La conversion vise ML Program avec un deployment target minimum d'iOS 15.
`compute_units` est stocké sur le modèle converti et peut être redéfini au moment où
l'artefact est chargé.

Les métadonnées du modèle vont dans `user_defined_metadata` sous forme de chaînes, et c'est là que le
backend lit la famille, la tâche, les noms de classe, la taille d'entrée et le schéma de pose.

### NMS embarqué

<code-tabs name="nms" />

`nms=True` enveloppe le modèle dans un pipeline Core ML qui se termine par la couche
`NonMaximumSuppression` d'Apple. Le résultat a deux sorties : `confidence`, de forme
`N` par le nombre de classes, et `coordinates`, de forme `N` par 4 en `xywh` normalisé.

Cela s'applique à la détection YOLOX et YOLO9 uniquement, et exige batch 1. Les familles de
style DETR sont refusées par leur nom, parce que la prédiction d'ensembles fait un top-k sur
les queries et les classes sans étape d'IoU et ne peut pas utiliser cette couche. `max_det` n'est pas
exposé ici non plus ; quand le plafond de détections compte, utilisez plutôt
[le NMS embarqué d'ONNX](/docs/export/onnx).

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` reconnaît un répertoire portant le suffixe `.mlpackage` et renvoie le
même objet `Results` que le checkpoint. `compute_units` est le seul argument que la
factory transmet pour ce format, et il accepte `all`, `cpu_and_gpu`,
`cpu_and_ne` et `cpu_only`. L'argument `device` est ignoré, parce que Core ML
passe par les compute units à la place.

Le second snippet est le chemin du runtime brut. Le letterboxing, le décodage, le NMS et
le rééchelonnement des coordonnées y sont à votre charge, et les noms de classe vivent dans
`user_defined_metadata`.

## Contraintes

Quatre familles, détection uniquement : `yolox`, `yolo9`, `rtdetr` et `rfdetr`. Tout le
reste est refusé au preflight, parce que le wrapper de prétraitement conscient de la famille est
ce qui rend correct le contrat d'entrée image fixe, et une famille en dehors de cet ensemble
convertirait avec la mauvaise normalisation. L'erreur nomme ONNX et TorchScript comme
alternatives.

La forme d'entrée est figée par `ct.ImageType`, donc `dynamic=True` ne change rien
et les métadonnées enregistrent `dynamic=False`. Exportez un second bundle pour une seconde
résolution.

`half=True` convertit en précision de calcul FP16. Il n'y a pas de chemin INT8 depuis cet
exporteur.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour le format on-device plus récent
d'Apple, voir [Core AI](/docs/export/coreai). Pour une seule combinaison :

<code-tabs name="support" />
