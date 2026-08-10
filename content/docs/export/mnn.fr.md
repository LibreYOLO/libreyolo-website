---
title: MNN
seo_title: "Exporter vers MNN depuis LibreYOLO"
description: "Exportez un détecteur LibreYOLO vers MNN via ONNX et mnnconvert : une forme NCHW fixe, du FP32 sur CPU, et un sidecar de métadonnées exigé par le contrat de runtime."
lead: "MNN est le moteur d'inférence léger d'Alibaba. LibreYOLO exporte un graphe ONNX statique, le convertit avec l'outil mnnconvert fourni par le paquet MNN, et écrit un sidecar JSON qui enregistre les noms d'entrée et de sortie, la forme d'entrée fixe et les noms de classe."
keywords:
  - exporter yolo mnn
  - mnnconvert
  - inférence mnn
  - détection d'objets sur mobile
  - forme nchw fixe
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="mnn")'
    mono: true
  - label: Écrit
    value: "Un fichier .mnn plus un sidecar de métadonnées .mnn.json"
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Se recharge avec
    value: 'LibreYOLO("weights/LibreYOLO9t.mnn")'
    mono: true
  - label: Formes
    value: "NCHW fixe. dynamic=True est refusé."
  - label: Précision
    value: "FP32 uniquement, CPU uniquement."
  - label: Tâches
    value: "Détection uniquement dans cette version"
verification: "Lu depuis libreyolo/export/mnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/mnn.py et pyproject.toml sur la branche dev."
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # L'extra inclut libreyolo[onnx], car MNN convertit depuis un ONNX intermédiaire.
        pip install "libreyolo[mnn]"
    - label: Vérifier que le convertisseur est dans le PATH
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.mnn et weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, ou (hauteur, largeur)
            batch=1,          # figé dans l'artefact
            simplify=True,    # onnxsim sur l'ONNX intermédiaire
            output_path=None, # None écrit weights/<stem>.mnn
            verbose=False,    # True diffuse le log de mnnconvert
        )

        # dynamic=True lève ValueError. half=True et int8=True sont refusés.
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN brut
      language: python
      code: |
        import json

        import MNN
        import numpy as np

        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))
        print(meta["mnn_input_names"], meta["mnn_output_names"], meta["mnn_input_shape"])

        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )
        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )

        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)
        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )
        outputs = module.forward([input_var])
        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Le prétraitement et le postprocessing sont à votre charge sur ce chemin.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installation

<code-tabs name="install" />

L'extra inclut `libreyolo[onnx]`, parce que la conversion passe par un ONNX
intermédiaire. Il apporte aussi l'exécutable `mnnconvert`, que l'exporteur cherche
d'abord à côté de l'interpréteur Python actif, puis dans le `PATH`. Un convertisseur
absent lève une `ImportError` qui nomme la commande d'installation plutôt que
d'échouer en pleine conversion.

## Export

<code-tabs name="export" />

Avant de passer le graphe, l'exporteur lit le contrat d'entrée ONNX et refuse tout ce
qu'il ne peut pas exprimer : plus d'une entrée image, ou une forme d'entrée avec une
dimension symbolique. Dans cette version, MNN exige une forme NCHW entièrement fixe, et
`batch` est figé dans l'artefact plutôt que négocié au chargement.

Le sidecar n'est pas de la comptabilité optionnelle. `weights/LibreYOLO9t.mnn.json`
enregistre les noms d'entrée et de sortie, la forme d'entrée fixe, le batch, les noms de
classe, la version de MNN utilisée et le backend pour lequel l'artefact a été construit,
et le runtime valide chacun de ces champs au chargement.

Sous Windows, MNN 3.6.1 termine parfois la conversion puis s'arrête pendant la
destruction du processus avec une violation d'accès ou un statut fail-fast. L'exporteur
reconnaît ces codes de sortie précis et considère la conversion comme réussie quand le
fichier de sortie est présent.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` s'oriente sur le suffixe `.mnn` et renvoie le même objet `Results` que le
checkpoint. Le chargement est strict par conception : le sidecar doit déclarer
`format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, une taille, une
tâche de détection, une forme NCHW fixe et positive qui concorde avec la taille d'image
enregistrée, et des noms de classe couvrant chaque indice de 0 à `nc - 1`. Toute
divergence lève au lieu de deviner.

Prédire à un `imgsz` différent de celui pour lequel l'artefact a été construit lève
aussi, et `device` est ignoré avec un avertissement, parce que les exports MNN
s'exécutent ici sur CPU.

Le second snippet est le chemin du runtime brut. Le prétraitement, le décodage, le NMS
et le rééchelonnement des coordonnées y sont à votre charge, et les noms d'entrée et de
sortie viennent du sidecar parce que le chargeur de modules de MNN les veut
explicitement.

## Contraintes

Détection uniquement. Le backend refuse toute autre tâche au chargement, et le côté
export suit : en dehors des combinaisons enregistrées, le preflight lève avec « MNN v1
has no implemented runtime contract for this family and task. »

FP32, CPU, forme fixe. `dynamic=True` lève `ValueError`, et `half=True` et `int8=True`
sont refusés pendant la validation.

Les familles de détection validées sont YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM et YOLO-NAS, chacune couverte par la
conversion, un rechargement de l'artefact neuf, une exécution MNN sur CPU, des
vérifications de métadonnées et une parité des détections post-NMS appariées avec le
modèle PyTorch. DEIMv2 convertit, se recharge, s'exécute et conserve les détections
post-NMS, mais sa route ONNX intermédiaire a une parité des scores au niveau des queries
incomplète, elle est donc enregistrée comme disponible plutôt que validée.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
