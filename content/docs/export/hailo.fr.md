---
title: Hailo
seo_title: Exécuter des modèles LibreYOLO sur les accélérateurs Hailo
description: "Déployez un modèle LibreYOLO sur un Hailo-8 ou un Hailo-8L\_: l'export ONNX statique, l'étape Dataflow Compiler que vous exécutez vous-même, et les architectures qui compilent."
lead: "Les accélérateurs Hailo se compilent avec le Hailo Dataflow Compiler, un SDK propriétaire distribué via la Developer Zone de Hailo. La part de LibreYOLO dans le flux est un simple export ONNX statique\_; le parsing, la quantification et la compilation vers un HEF se font ensuite dans le DFC."
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - compiler un hef
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Étape LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Pas un format
    value: >-
      Il n'existe pas de format="hef". Le DFC ne peut pas être une dépendance
      pip.
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Hôte de compilation
    value: >-
      Linux x86_64, y compris WSL2 Ubuntu 22.04. La compilation ne peut pas
      s'exécuter sur ARM.
  - label: Compile
    value: >-
      Graphes purement CNN, à formes fixes. L'attention, les formes dynamiques
      et les architectures dominées par LayerNorm, non.
  - label: Statut
    value: >-
      Aucune famille LibreYOLO n'a encore été menée de bout en bout dans le DFC
      jusqu'à un HEF fonctionnel.
verification: "Lu depuis skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py et libreyolo/cli/commands/export.py sur la branche dev. Les contraintes du DFC sont celles consignées dans ce skill\_; aucun HEF LibreYOLO n'a été compilé ni mesuré."
snippets:
  install:
    - label: Côté LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Côté Hailo, installé par vos soins'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo exige un batch 1, une résolution fixe et aucun axe dynamique.
        # dynamic=True est le défaut de l'API Python : désactivez-le vous-même.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # La CLI utilise déjà des formes statiques par défaut.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Vérifier que le graphe est statique avant de compiler
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Parser, quantifier et compiler'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Pour YOLOX, traduisez une fois sans end_node_names : le log du DFC

        # affiche les end nodes qu'il suggère. Relancez avec ceux-là.

        runner.translate_onnx_model(ONNX)


        # La normalisation doit correspondre au prétraitement LibreYOLO. YOLOX

        # et YOLO9 n'ont besoin ni de moyenne ni d'écart type, seulement de la

        # mise à l'échelle de 0-255 vers 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Facultatif : laissez Hailo gérer le NMS. La configuration dépend à la

        # fois du nombre de classes et de la taille d'entrée, donc une config

        # COCO-80 est fausse pour un modèle à trois classes affiné. Sans cette

        # ligne, le HEF émet les tenseurs bruts de la tête et l'application

        # les décode.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Les images de calibration doivent être représentatives des données de

        # déploiement. Des images aléatoires compilent et détruisent

        # silencieusement l'exactitude.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: End nodes de YOLO9
      language: python
      code: |
        # Les graphes LibreYOLO utilisent un préfixe "/head/...", pas le préfixe
        # "model.N" vu dans les configurations écrites pour d'autres exports.
        # Une config copiée ne correspondra pas. Vérifiez les noms dans votre
        # propre graphe si le parsing échoue.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 avec l'AI Kit ou l'AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # test de l'appareil, il nomme
        l'arch

        hailortcli run libreyoloxs.hef       # smoke test et débit
source_hash: 33b077f1c23d5535
---

## Installation

Il n'existe pas de `format="hef"` dans LibreYOLO, et il n'y en aura pas. Le Hailo
Dataflow Compiler est un SDK propriétaire distribué sous forme de wheel privée
derrière une inscription à la Developer Zone : il ne peut donc être ni une
dépendance ni un extra. Le déploiement se fait en deux étapes : LibreYOLO écrit un
fichier ONNX statique, et vous exécutez le DFC dessus.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Export

<code-tabs name="export" />

Ne passez pas `half=True`. Le DFC ingère de l'ONNX FP32 et fait sa propre
quantification INT8. Ne passez pas non plus `nms=True` : soit Hailo gère le NMS
via `nms_postprocess`, soit l'application s'en charge, et un sous-graphe NMS est du
poids mort au-delà des end nodes. L'opset par défaut fonctionne ; si le parser du
DFC proteste, ré-exportez avec `opset=11`.

Le DFC coupe le graphe aux end nodes que vous fournissez, à savoir les convolutions
de la tête de détection, et jette tout ce qui se trouve en aval. L'ONNX décodé
ordinaire de LibreYOLO est donc une entrée acceptable : la queue de décodage est
simplement ignorée par le parser.

## Compilation

<code-tabs name="compile" />

Choisissez `hw_arch` selon la cible : `hailo8` pour le Hailo-8, l'AI HAT+ 26 TOPS
et les modules M.2 et PCIe ; `hailo8l` pour le Hailo-8L, le Raspberry Pi AI Kit et
l'AI HAT+ 13 TOPS ; `hailo10h` pour le Hailo-10H, qui exige un DFC et un Model Zoo
plus récents et assortis. En cas de doute, `hailortcli fw-control identify` sur
l'appareil répond à la question.

Deux familles correspondent à une méta-architecture NMS de HailoRT, ce qui permet à
Hailo de gérer la suppression à l'intérieur du pipeline compilé : YOLOX via
`meta_arch=yolox`, et YOLO9 via la méta-architecture à tête découplée de Hailo,
dont la disposition de tête est identique. Prenez la configuration
`nms_postprocess` correspondante dans le Hailo Model Zoo et ajustez-la à votre
nombre de classes et à votre taille d'entrée. Tout autre détecteur convolutif
compile comme un graphe sans méta-architecture correspondante : le HEF émet les
tenseurs bruts de la tête et l'application exécute le décodage et le NMS sur le CPU.

Conservez le log de compilation quand quelque chose échoue. Chaque correctif tient
au nom exact de la couche ou de l'opérateur qui échoue.

## Exécuter l'artefact

<code-tabs name="device" />

L'inférence applicative passe par l'API Python `hailo_platform`. Avec
`nms_postprocess` compilé dedans, la sortie est `(batch, num_classes, max_dets, 5)`
et porte `[y1, x1, y2, x2, score]` en coordonnées du modèle, que vous remettez
vous-même à l'échelle de l'image source. Le pipeline `Results` de LibreYOLO
n'intervient pas à l'exécution ; le HEF est un artefact autonome, et le
prétraitement comme le post-traitement appartiennent à l'application.

## Contraintes

Qu'un modèle puisse ou non cibler le Hailo-8 ou le Hailo-8L est une propriété de
son architecture, pas de son nom ; la règle ci-dessous vaut donc aussi pour les
familles ajoutées après la rédaction de cette page.

Un modèle ne compilera pas s'il contient l'un de ces éléments :

- De l'attention, quelle qu'elle soit : self, cross, déformable ou fenêtrée. Cela
  exclut tout détecteur de type DETR, tout détecteur à vocabulaire ouvert ou
  conditionné par du texte, tout backbone ViT, et toute tour langage ou
  vision-langage. Le zoo de Hailo lui-même propose quelques HEF de transformers
  réglés à la main ; c'est du travail sur mesure du fournisseur, et ce n'est pas la
  preuve qu'un graphe d'attention quelconque compile.
- Des formes dynamiques ou un flot de contrôle dépendant des données. Le DFC
  compile une seule forme d'entrée fixe et un graphe statique, donc les nombres de
  requêtes variables, les prompts textuels, le top-k dynamique, `NonZero`, `Gather`
  ou `TopK` à indices dynamiques et `grid_sample` sont tous exclus.
- Une architecture dominée par LayerNorm ou par GELU. La BatchNorm se replie
  proprement dans les convolutions ; la prise en charge de LayerNorm est mauvaise et
  GELU n'est pas une activation native, si bien qu'une pile de type ConvNeXt
  convient mal, même si elle est nominalement convolutive.
- Du travail image-à-image à la résolution native. Les modèles de restauration
  tournent à pleine résolution d'entrée et dépassent les budgets SRAM praticables
  d'un Hailo.

Une famille est candidate lorsqu'elle est purement convolutive, utilise de la
BatchNorm avec ReLU ou SiLU, et a une taille d'entrée fixe. Dans cette
bibliothèque, cela recouvre les détecteurs CNN à un étage, avec YOLOX et YOLO9
comme cibles principales ; les autres détecteurs convolutifs tels que PicoDet,
YOLO-NAS et RTMDet, avec décodage côté application ; les classifieurs CNN ResNet,
MobileNetV4-conv et EfficientNetV2, dont ResNet est le mieux pris en charge parce
que le Model Zoo de Hailo fournit des recettes pour lui ; et les petites têtes de
tâche convolutives comme la détection de points FOMO et l'estimation du regard L2CS
sur un backbone ResNet, compilables en principe mais sans recette Hailo.

Une réserve sur le statut, qui est la raison pour laquelle rien sur cette page
n'est présenté comme pris en charge : aucune famille LibreYOLO n'a été menée de
bout en bout dans le DFC jusqu'à un HEF fonctionnel. Les règles ci-dessus prédisent
la compilabilité à partir de l'architecture. Le comportement du parser, la
quantification et l'exactitude restent non prouvés tant qu'un HEF n'a pas été
compilé et mesuré. Traitez donc chaque candidat comme exigeant ses propres preuves
consignées : un HEF compilé à partir du checkpoint exact, avec les versions du DFC,
du Model Zoo et de HailoRT enregistrées, une calibration documentée, et une
comparaison d'exactitude sur l'appareil face à la référence FP32 plutôt qu'un
chiffre de débit.

Si le modèle est disqualifié, les solutions de repli sont les runtimes à parité
consignée : [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) et
[OpenVINO](/docs/export/openvino).
