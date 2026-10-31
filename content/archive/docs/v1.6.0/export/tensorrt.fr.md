---
title: TensorRT
seo_title: Exporter vers TensorRT depuis LibreYOLO
description: "Construisez un moteur TensorRT à partir d'un modèle LibreYOLO\_: l'intermédiaire ONNX, les builds FP16 et INT8, les profils de batch dynamique et les limites de portabilité des moteurs."
lead: >-
  TensorRT compile un graphe en un moteur optimisé pour un GPU donné. LibreYOLO
  exporte d'abord un intermédiaire ONNX, le parse avec le parseur ONNX de
  TensorRT, construit le moteur et écrit les métadonnées du modèle à côté, dans
  un fichier annexe JSON.
keywords:
  - exporter yolo tensorrt
  - moteur tensorrt
  - trt fp16
  - calibration int8 tensorrt
  - profil d'optimisation tensorrt
  - batch dynamique tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tensorrt")
    mono: true
  - label: Écrit
    value: Un fichier .engine plus un fichier annexe de métadonnées .engine.json
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Formes
    value: "Statiques par défaut\_; dynamic=True ajoute un profil d'optimisation sur l'axe batch"
  - label: Précision
    value: 'FP32, FP16 (half=True), INT8 (int8=True avec data=)'
  - label: Prérequis
    value: >-
      Un GPU NVIDIA à la construction comme à l'exécution. Les moteurs ne
      passent pas d'une architecture GPU à une autre.
verification: >-
  Lu depuis libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py et pyproject.toml
  sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: >
        # Le moteur est construit depuis un intermédiaire ONNX, d'où les deux
        extras.

        pip install "libreyolo[onnx,tensorrt]"
    - label: Vérifier la toolchain avant de construire
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Écrit weights/LibreYOLO9t_fp16.engine et
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Arguments
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # requis quand int8=True
            dynamic=False,
            workspace=4.0,                  # GiB de mémoire de travail au build
            min_batch=1,                    # bornes du profil dynamique
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # ou "ampere_plus"
            gpu_device=0,                   # GPU de build sur une machine multi-GPU
            verbose=False,
        )
  dynamic:
    - label: Moteur à batch dynamique
      language: python
      code: |
        from libreyolo import LibreYOLO

        # L'intermédiaire ONNX doit porter l'axe batch dynamique pour que le
        # profil ait quelque chose à quoi se rattacher.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 avec données de calibration
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # requis, aucune valeur par défaut ici
            fraction=1.0,
        )
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT brut
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Noms de classes, tâche et taille d'entrée sont dans l'annexe, pas le
        moteur.

        # L'allocation des buffers, le pré et le post-traitement sont à votre
        charge.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Vérifier une famille et une tâche avant de construire
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Installation

La construction comme l'exécution demandent un GPU NVIDIA avec une pile CUDA
fonctionnelle. Il n'y a pas de repli CPU pour ce format.

<code-tabs name="install" />

L'extra `tensorrt` épingle `tensorrt-cu12` et `pycuda`, et le marqueur écarte les
deux sur macOS. Sur un Jetson, n'utilisez pas cet extra : il épingle une build
CUDA 12 face à une plateforme CUDA 13. Utilisez plutôt le TensorRT installé par
JetPack, comme décrit sur [NVIDIA Jetson](/docs/export/jetson).

## Export

<code-tabs name="export" />

L'export se déroule en deux étapes. La première écrit un intermédiaire ONNX dans
un chemin temporaire, la seconde le parse et construit le moteur, et
l'intermédiaire est supprimé ensuite. `workspace` est la mémoire de travail du
build, en GiB ; une valeur plus grande laisse le builder essayer plus de kernels
et n'affecte pas la mémoire d'inférence.

Le fichier annexe de métadonnées est écrit à côté du moteur sous le nom
`<engine>.json` et consigne la précision que le build a réellement obtenue. Quand
le GPU n'a pas de FP16 rapide ou d'INT8 rapide, le builder avertit et se replie,
et le fichier annexe indique la précision obtenue plutôt que celle demandée.

En FP16, un backbone ViT présent dans le graphe est détecté et ses couches
flottantes sont épinglées en FP32. Les backbones de type DINOv2 débordent en FP16
et produisent des NaN, donc le build active `OBEY_PRECISION_CONSTRAINTS` et
rapporte `FP16 (FP32 ViT backbone)`. La passe ne fait rien sur les backbones CNN.

### Batch dynamique

<code-tabs name="dynamic" />

`dynamic=True` ajoute un profil d'optimisation allant de `min_batch` à
`max_batch`, optimisé à `opt_batch`, et consigne ces trois valeurs dans le fichier
annexe. Le profil n'est ajouté que si l'intermédiaire ONNX porte effectivement une
dimension de batch dynamique ; sinon le build journalise qu'il utilise une
optimisation statique et poursuit.

### INT8

<code-tabs name="int8" />

L'INT8 utilise le calibrateur d'entropie de TensorRT au-dessus d'un loader de
calibration LibreYOLO, et `data` est obligatoire : ce format n'a pas de repli sur
huit images. La calibration a besoin de `cuda-python` ou de `pycuda` pour le
buffer côté device. Le cache de calibration est indexé sur un hash des octets
ONNX, si bien que les échelles d'un modèle ne sont jamais réutilisées pour un
autre qui écrirait au même chemin de sortie.

`half=True` et `int8=True` ensemble déclenchent un avertissement et construisent
en INT8, ce qui conserve un repli FP16 pour les couches que TensorRT ne peut pas
quantifier.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` s'aiguille sur le suffixe `.engine`, lit le fichier annexe pour les
noms de classes, la tâche et le schéma de pose, et renvoie le même objet `Results`
que le checkpoint. Il lève une erreur immédiatement si aucun périphérique CUDA
n'est présent.

Le second extrait est la voie du runtime brut. L'allocation des buffers hôte et
device, le prétraitement, le décodage, le NMS et la remise à l'échelle des
coordonnées passent tous à votre charge, et le moteur lui-même ne porte aucun nom
de classe : le fichier annexe doit donc l'accompagner.

## Contraintes

Un moteur sérialisé est lié à l'architecture GPU, à la pile de pilotes et à la
version de TensorRT qui l'a construit. Un moteur construit sur une station de
travail ne se chargera pas sur une architecture différente, et c'est pourquoi
l'étape de build tourne sur la machine de déploiement.
`hardware_compatibility="ampere_plus"` échange un peu de performance contre la
portabilité sur Ampere et plus récent. La valeur `"same_compute_capability"`
correspond à `NONE` et déclenche un avertissement : le moteur n'est optimisé que
pour le GPU courant, et l'export le dit plutôt que de revendiquer une portabilité
qu'il n'a pas appliquée.

Seul l'axe batch fait l'objet d'un profil. Un build avec des dimensions spatiales
dynamiques ne fait pas partie de ce contrat, et c'est pourquoi FCOS est bloqué :
il lui faut une hauteur et une largeur paddées dynamiques pour préserver sa
transformation d'aspect 800 par 1333.

Bloqués avant le tracing : la segmentation YOLO9, la segmentation RTMDet-Ins, la
détection SSD, Faster R-CNN et RetinaNet, et le matting BiRefNet ou FeyNobg, où
TensorRT 10.16 atteint le nœud ONNX `DeformConv` partagé et ne peut pas le parser
parce que `ModulatedDeformConv2d` est absent du registre de plugins.

Lorsqu'une combinaison n'est ni validée ni bloquée, le chemin de conversion est
disponible et le projet n'a pas enregistré de parité au runtime TensorRT pour
elle. C'est un constat sur les preuves disponibles, pas sur la réussite du build.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une combinaison
précise :

<code-tabs name="support" />
