---
title: TFLite
seo_title: Exporter vers TFLite (LiteRT) depuis LibreYOLO
description: "Exportez un modèle LibreYOLO vers un FlatBuffer .tflite via onnx2tf\_: formes statiques, FP32 uniquement, entrées NHWC, et les familles qui se convertissent proprement."
lead: >-
  TFLite est le format FlatBuffer que LiteRT exécute sur les cibles mobiles et
  embarquées. LibreYOLO exporte un graphe ONNX statique, le convertit avec
  onnx2tf en mode flatbuffer-direct, et écrit les métadonnées du modèle à côté
  de l'artefact sous forme de sidecar JSON.
keywords:
  - exporter yolo tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - flatbuffer tflite
  - entrée nhwc tflite
  - inférence sur appareil edge
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tflite")
    mono: true
  - label: Écrit
    value: Un fichier .tflite plus un sidecar de métadonnées .tflite.json
  - label: Extra
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Formes
    value: Statiques uniquement. dynamic=True est refusé.
  - label: Précision
    value: FP32 uniquement. half=True et int8=True sont refusés.
  - label: Prérequis
    value: >-
      Python 3.12 ou plus récent, car onnx2tf 2.4.x ne publie pas de wheels plus
      anciennes
verification: >-
  Lu depuis libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py et pyproject.toml
  sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # LiteRT est le nom actuel de TensorFlow Lite chez Google. Les deux
        # extras installent la même toolchain et produisent le même .tflite.
        pip install "libreyolo[tflite]"
    - label: Vérifier d'abord la version de Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.tflite et weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" est accepté comme alias et pointe vers le même exporteur.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Arguments
      language: python
      code: >
        model.export(
            format="tflite",
            imgsz=640,        # int, ou (hauteur, largeur)
            batch=1,
            simplify=True,    # onnxsim sur l'ONNX intermédiaire
            output_path=None, # None écrit weights/<stem>.tflite
            verbose=False,    # True affiche le log onnx2tf
        )


        # dynamic=True lève ValueError, le convertisseur exige des formes
        statiques.

        # half=True et int8=True sont refusés avant le tracing.
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT seul
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, et non NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Les noms de classe, la tâche et la taille d'entrée sont dans le
        sidecar.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Prétraitement, transposition NCHW vers NHWC et post-traitement sont à
        vous.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Installation

<code-tabs name="install" />

L'extra installe `onnx2tf` pour la conversion et `ai-edge-litert` pour exécuter le
résultat, les deux derrière un marqueur Python 3.12. Sur un interpréteur plus
ancien, l'export lève une `ImportError` qui nomme l'exigence de version au lieu
d'échouer à l'intérieur du convertisseur.

`libreyolo[litert]` installe exactement la même chose. La chaîne de format `litert`
est un alias de `tflite`, et le fichier de sortie est un `.tflite` dans les deux cas.

## Export

<code-tabs name="export" />

La famille et la tâche sont vérifiées avant toute autre chose, si bien qu'une
combinaison non prise en charge échoue immédiatement avec l'erreur précise du
convertisseur ou du runtime qui l'a écartée, et non un message générique. La
conversion elle-même est un appel en sous-processus à `onnx2tf` en mode
`flatbuffer_direct` sur un ONNX intermédiaire statique.

Les métadonnées sont un sidecar. `weights/LibreYOLO9t.tflite.json` porte la
famille, la tâche, les noms de classe, la taille d'entrée et le schéma de pose ;
le FlatBuffer lui-même n'a aucun champ de métadonnées LibreYOLO, les deux fichiers
voyagent donc ensemble.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` aiguille sur le suffixe `.tflite` et renvoie le même objet `Results`
que le checkpoint. Le backend lit le sidecar, transpose le blob NCHW en NHWC
lorsque l'interpréteur demande une entrée channels-last, applique l'échelle de
quantification et le zero point de l'interpréteur lorsqu'ils sont présents, et
retranspose les sorties dans la disposition qu'attend le post-traitement de
LibreYOLO.

Le second snippet est la voie du runtime seul. Le prétraitement, la transposition
de disposition, le décodage, la NMS et la remise à l'échelle des coordonnées y
deviennent tous à votre charge, et le détail de disposition est celui qui passe le
plus souvent inaperçu : onnx2tf produit des entrées channels-last, un blob de
forme `(1, 3, 640, 640)` ne se liera donc pas.

## Contraintes

Formes statiques uniquement. `dynamic=True` lève une `ValueError` avant le tracing,
et le canevas d'export est figé à la valeur résolue par `imgsz`.

FP32 uniquement. `half=True` et `int8=True` sont tous deux refusés pendant la
validation, le déploiement quantifié n'est donc pas accessible depuis cet
exporteur aujourd'hui.

La couverture est plus étroite ici que pour les formats de graphe, et elle est
décidée par la mesure plutôt que par famille. Les combinaisons validées comprennent
la détection YOLO9, YOLOX et YOLO-NAS, la segmentation sémantique PIDNet, les
quatre familles de classification CNN, l'embedding DINOv2 et SigLIP2, la
classification SigLIP2, la détection de contours TEED et DexiNed, et la
restauration Real-ESRGAN et SwinIR. SwinIR s'accompagne d'une réserve
supplémentaire : la parité tient quand les dimensions de la source correspondent
exactement au canevas d'export, et les sources plus petites sont complétées jusqu'au
canevas avant l'exécution du transformer, ce qui peut diverger de l'inférence native
à taille variable.

Les entrées bloquées nomment la défaillance exacte, ce qui vaut la peine d'être lu
avant de tenter un contournement. Quelques exemples : la détection RF-DETR se
convertit sur son canevas natif de 384 mais LiteRT ne peut pas l'allouer, car
`STRIDED_SLICE` reçoit une entrée au-delà du rang 5-D qu'il prend en charge ;
PicoDet est refusé parce qu'un `RESHAPE` fait correspondre 19 200 éléments d'entrée
à 9 600 éléments de sortie ; D-FINE fait planter le convertisseur dans la gestion
des formes de `GatherElements` ; RTMDet s'exporte et se recharge avec une parité
brute intacte, mais les boîtes publiques tombent à 0.911 IoU avec 29.9 px de dérive
de coordonnées.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison,
y compris la chaîne de raison derrière un blocage :

<code-tabs name="support" />
