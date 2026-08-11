---
title: Paddle
seo_title: Exporter vers PaddlePaddle depuis LibreYOLO
description: "Convertissez un détecteur LibreYOLO en modèle d'inférence PaddlePaddle via X2Paddle\_: la chaîne d'outils épinglée, les graphes statiques FP32 en batch 1, et l'inférence sur CPU."
lead: >-
  Un modèle d'inférence PaddlePaddle, c'est un graphe model.pdmodel accompagné
  d'un fichier de poids model.pdiparams. LibreYOLO exporte un graphe ONNX
  statique en opset 15, le convertit avec X2Paddle et empaquette le résultat
  avec un metadata.yaml pour qu'il se charge par la même factory que tous les
  autres runtimes.
keywords:
  - exporter yolo paddle
  - inférence paddlepaddle
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Écrit
    value: 'Un répertoire contenant model.pdmodel, model.pdiparams et metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Se recharge avec
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Formes
    value: 'Statiques, batch 1, opset 15. Les trois sont imposés.'
  - label: Précision
    value: 'FP32 uniquement, CPU uniquement.'
  - label: Chaîne d'outils
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 ou antérieur, vérifiés
      exactement
verification: >-
  Lu depuis libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md et
  pyproject.toml sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: >
        # Python 3.10 à 3.12. WSL2 avec Ubuntu 22.04 est le chemin Windows
        validé.

        pip install "libreyolo[paddle]"
    - label: Vérifier les versions épinglées
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit le répertoire weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Arguments
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int, le canevas carré de cette famille
            batch=1,          # toute autre valeur lève ValueError
            dynamic=False,    # True lève ValueError
            simplify=True,    # False lève ValueError
            opset=15,         # toute autre valeur lève ValueError
            output_path=None, # None écrit weights/<stem>_paddle
        )
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Le backend directement
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Ce que LibreYOLO() construit pour un répertoire Paddle. Le même objet
        # Results, sans routage par la factory entre les deux.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle brut
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Le prétraitement et le postprocessing sont à votre charge ici.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Installation

<code-tabs name="install" />

L'extra épingle exactement la pile sur laquelle la parité a été mesurée :
PaddlePaddle 2.6.2, X2Paddle 1.6.0 et ONNX 1.17 ou antérieur. Ces épinglages sont
vérifiés au moment de l'export, pas seulement à l'installation, et une version
différente lève une `ImportError` qui nomme celle attendue. Les versions plus
récentes de Paddle rejettent une partie du code statique que génère X2Paddle
1.6.0 ; échouer tôt vaut donc mieux que produire un artefact que personne n'a
validé.

## Export

<code-tabs name="export" />

Quatre arguments sont figés plutôt que dotés d'une valeur par défaut. `dynamic`
doit valoir `False`, `batch` doit valoir 1, `simplify` doit valoir `True` pour
obtenir un graphe de conversion entièrement statique, et `opset` doit valoir 15,
le plafond qu'accepte X2Paddle 1.6.0. Passer autre chose lève avant le tracing.

Une seule normalisation s'applique au graphe intermédiaire. ONNX définit une
dilatation MaxPool omise comme valant un, PyTorch écrit explicitement l'attribut
rempli de uns, et X2Paddle 1.6.0 le rejette : l'exporteur supprime donc cette
valeur par défaut redondante et laisse l'opération spécifiée inchangée.

L'artefact est un répertoire : `model.pdmodel`, `model.pdiparams` et
`metadata.yaml`. Le code Python que X2Paddle génère pendant la conversion n'en
fait pas partie.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` reconnaît tout répertoire contenant à la fois `model.pdmodel` et
`model.pdiparams`, lit `metadata.yaml` et renvoie le même objet `Results` que le
checkpoint. Un `device` autre que `auto` ou `cpu` lève : ce backend fonctionne
uniquement sur CPU.

Ce que la factory construit, c'est `PaddleBackend`, exporté depuis `libreyolo` et
importable sous `libreyolo.backends.paddle.PaddleBackend`. Construisez-le
vous-même quand vous voulez le backend sans le routage par suffixe de la factory,
par exemple pour passer `task=` explicitement sur un répertoire dont vous n'avez
pas écrit le `metadata.yaml`. Son `predict()` accepte les mêmes sources et renvoie
les mêmes résultats.

Le snippet en runtime brut reprend ce que configure le backend, et les trois
options désactivées le sont délibérément. Le pipeline de fusion CPU de Paddle 2.6
peut planter en optimisant les grands graphes gather et scatter émis pour
l'attention déformable ; c'est donc sur le graphe statique portable et non fusionné
que la parité a été mesurée. Le prétraitement, le décodage, le NMS et le
rééchelonnement des coordonnées sont à votre charge sur ce chemin.

## Contraintes

Pas de formes dynamiques, pas de FP16, pas d'INT8, pas de NMS intégré, pas de
runtime GPU.

Les combinaisons validées sont la détection YOLO9, la détection YOLO9-E2E et
YOLO9-P2, la détection, l'estimation de pose et la segmentation EC, la détection
RT-DETRv4, D-FINE, DEIM et DEIMv2, et la détection et l'estimation de pose
YOLO-NAS. Chacune est couverte par la conversion, un rechargement du runtime sur
CPU, une parité des sorties brutes et des résultats publics appariés.

Bloquées, avec la raison enregistrée pour chaque combinaison :

| Combinaison | Pourquoi |
|---|---|
| RF-DETR, toutes tâches | Exige l'opset ONNX 17 et GridSample ; X2Paddle 1.6.0 accepte l'opset 15 ou inférieur et n'a pas de mapper GridSample |
| Détection RT-DETR et RT-DETRv2 | Les graphes entraînés exigent GridSample à l'opset 16 ou plus récent |
| Segmentation D-FINE | Se convertit et se recharge, mais l'erreur RMS relative des logits de masque est de 3.52 % et l'IoU minimal des masques appariés est de 0.582 |
| Segmentation YOLO9 | YOLO9 ne fait que de la détection dans LibreYOLO |
| Segmentation RTMDet-Ins | Le décodage de masque à noyaux dynamiques n'a pas de contrat de runtime exporté |

Tout ce qui n'est listé ni comme validé ni comme bloqué est refusé, avec la note
indiquant que cela n'a pas été validé via le chemin de conversion ONNX vers
Paddle.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
