---
title: NVIDIA DeepStream
seo_title: Exécuter des modèles YOLO sur NVIDIA DeepStream
description: "Exportez un modèle LibreYOLO pour NVIDIA DeepStream\_: un graphe ONNX plus une configuration nvinfer générée. Les commandes exactes pour la compilation du parser et pour le pipeline."
lead: >-
  NVIDIA DeepStream exécute l'inférence via son élément nvinfer, qui a besoin
  d'un graphe ONNX, d'un fichier de configuration correspondant et d'un parser
  de bounding box. Définir deepstream=True sur l'export ONNX écrit les deux
  premiers et les relie au troisième.
keywords:
  - NVIDIA DeepStream
  - DeepStream YOLO
  - nvinfer
  - parser bounding box DeepStream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - moteur TensorRT
  - yolo sur Jetson
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Écrit
    value: 'Un graphe ONNX, config_infer_primary_<stem>.txt et <stem>_labels.txt'
  - label: Couverture
    value: 43 combinaisons de famille et de tâche réparties sur neuf tâches
  - label: Parser
    value: >-
      NvDsInferParseYolo, issu du projet DeepStream-Yolo sous licence MIT de
      Marcos Luciano. À compiler une fois par appareil.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Disponibilité
    value: >-
      Livré dans la v1.5.0. Fusionné dans dev le 2026-08-08 dans la pull request
      728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Runtime validé
    value: 'DeepStream 8.0.0 sur une RTX 5070 Ti, détection uniquement, 2026-08-08'
verification: >-
  Écrit à partir de la validation runtime du 2026-08-08. Les listes de familles,
  les clés de configuration et les valeurs par défaut sont lues dans
  libreyolo/export/deepstream.py et libreyolo/export/exporter.py au commit
  5f81e11e, fusionné dans dev le même jour dans la pull request 728.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Écrit libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # et libreyolo9s_labels.txt dans le répertoire courant.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Gardez chaque modèle de détection dans son répertoire, toutes les

        # configs de détection nomment le même cache d'engine. Voir "Pièges
        connus".

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Arguments
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True est refusé pour tout autre format
            deepstream=True,
            conf=0.25,         # alimente pre-cluster-threshold (et classifier-threshold,
                               # segmentation-threshold sur ces tâches)
            iou=0.45,          # alimente nms-iou-threshold, omis avec cluster-mode=4
            batch=1,           # alimente batch-size et le nom du cache d'engine
            half=False,        # True marque la config en network-mode=2 (build fp16)
            int8=False,        # True marque la config en network-mode=1
            dynamic=True,      # axe de batch dynamique dans le graphe ONNX
            imgsz=640,         # alimente infer-dims=3;H;W
        )


        # deepstream=True et nms=True s'excluent mutuellement, DeepStream
        applique

        # la suppression à l'étape de clustering, rien n'est donc intégré au
        graphe.
    - label: Télécharger d'abord les poids D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Vérifier le passthrough GPU avant toute chose
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, à lancer dans le conteneur DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 sur cette image est un stub et la compilation
        échoue

        # avec "fatal error: crt/host_defines.h: No such file or directory".
        Résolvez

        # un toolkit qui porte vraiment l'en-tête. Sur l'image 8.0, c'est
        cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # L'image fournit libcublas.so.12 et libcublas.so.12.8.4.1 mais pas le

        # libcublas.so sans version exigé par -lcublas, donc l'édition de liens

        # échoue avec "/usr/bin/ld: cannot find -lcublas". Donnez au linker ces
        noms.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: La segmentation d'instances utilise un autre parser
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Lancer
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Les deux étapes dans un seul conteneur
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Disponibilité

L'export DeepStream est livré dans la v1.5.0. Il a été fusionné dans `dev` le
2026-08-08 dans la pull request 728, donc une installation à jour en dispose et
aucun épinglage de branche n'est nécessaire.

<code-tabs name="install" />

Si vous avez cloné la branche `deepstream-export` avant le 2026-08-08,
remplacez-la. Cette branche a été rebasée et force-pushée, et l'ancien
historique n'a pas le correctif sans lequel ces exports ne tournent pas du tout
sur une machine CUDA.

## Ce que l'export écrit

`model.export(format="onnx", deepstream=True)` écrit trois fichiers côte à côte.
Pour `libreyolo9s.pt` :

- `libreyolo9s.onnx`, le graphe de détection, un seul tenseur de sortie de forme
  `(batch, num_detections, 6)`, chaque ligne `[x1, y1, x2, y2, score, class_id]`
  en coordonnées pixel de l'entrée réseau.
- `config_infer_primary_libreyolo9s.txt`, une configuration `nvinfer` qui porte
  les constantes de prétraitement de la famille, le nombre de classes, les seuils
  et le câblage du parser.
- `libreyolo9s_labels.txt`, un nom de classe par ligne.

Un fichier d'étiquettes apparaît dès que le checkpoint porte des noms de classes.
Les modèles de profondeur n'en ont pas : ils n'obtiennent ni le fichier ni la clé
`labelfile-path`.

LibreYOLO n'émet pas de `.so`. Le `.so` que DeepStream charge est le parser de
bounding box de `marcoslucianops/DeepStream-Yolo`, compilé une fois par appareil,
et c'est le même binaire quel que soit le détecteur LibreYOLO que vous lui donnez.
Le modèle, c'est l'ONNX. La classification et la segmentation sémantique n'ont
besoin d'aucun parser, parce que `nvinfer` les post-traite lui-même.

## Exporter le modèle

<code-tabs name="export" />

`LibreDFINE._load_weights` lève `FileNotFoundError` quand le fichier n'est pas
déjà sur le disque, sans tenter de téléchargement : récupérez donc
`LibreDFINEs.pt` vous-même au préalable. Ce manque est suivi dans
[l'issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Les poids
YOLO9 se téléchargent à la première utilisation.

Le flag n'existe qu'en Python. `libreyolo export` sur cette branche n'a pas
d'option `deepstream`, et la CLI construit ses arguments d'export à partir d'une
liste fixe au lieu de laisser passer les clés inconnues.

## Compiler le parser de bounding box

La détection a besoin de la bibliothèque parser, la segmentation d'instances en
demande une autre, et les autres tâches n'en demandent aucune. Deux points de
l'image DeepStream 8.0 cassent la commande de compilation documentée, et les deux
relèvent de l'environnement plutôt que de LibreYOLO.

L'image fournit `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` et `cuda-12.9` sous
`/usr/local`. Seul `cuda-12.5` a un toolkit complet. Elle fournit aussi
`libcublas.so.12` et `libcublas.so.12.8.4.1` mais pas le `libcublas.so` sans
version contre lequel `-lcublas` se résout. Le script ci-dessous contourne les
deux.

<code-tabs name="parser" />

Faites ensuite pointer `custom-lib-path` dans la config générée vers le
`libnvdsinfer_custom_impl_Yolo.so` compilé. La valeur générée est le chemin
relatif `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, qui se
résout quand `deepstream-app` tourne depuis le checkout `DeepStream-Yolo` et
qu'il faut modifier sinon.

## Lancer le pipeline

Vérifiez que le conteneur voit bien le GPU avant de passer du temps sur autre
chose. C'est le contrôle que la validation a fait en premier, sur une carte
Blackwell sous WSL2.

<code-tabs name="gpu" />

La validation a piloté `deepstream-app` avec une source fichier, sans sink
d'affichage, l'affichage à l'écran activé, et `gie-kitti-output-dir` défini pour
que les détections de chaque frame atterrissent sur le disque au format texte
KITTI. Une config avec ces réglages :

<code-tabs name="run" />

`nvinfer` construit l'engine TensorRT à partir de l'ONNX au premier lancement et
le met en cache à côté du modèle : le premier lancement paie la construction de
l'engine, les suivants chargent le cache.

## La configuration générée

Les deux configs ci-dessous ont été écrites par l'exporteur pour la validation,
sans retouche ensuite.

| Clé | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Les deux configs diffèrent en trois points : `maintain-aspect-ratio`,
`cluster-mode`, et la présence même de `nms-iou-threshold`. La config de D-FINE
omet complètement cette clé, ce qu'exige `cluster-mode=4`.

Les têtes qui émettent au plus une prédiction par objet reçoivent
`cluster-mode=4`, si bien que DeepStream ne fait aucun clustering dessus ; le
clustering fusionnerait des détections réellement distinctes. Cela couvre
`rfdetr`, `dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` et
`yolo9_e2e`. Les têtes à grille et à ancres reçoivent `cluster-mode=2` plus
`nms-iou-threshold`.

Les configs de détection portent aussi
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, qui confie la construction
de l'engine à la bibliothèque parser. C'est ce qui fige le nom du fichier de
cache d'engine, et c'est la source de la collision décrite dans les pièges
connus.

## Tâches et familles prises en charge

Quarante-trois combinaisons de famille et de tâche s'exportent.
`deepstream_supported_tasks()` et `deepstream_supported_families(task)` dans
`libreyolo/export/deepstream.py` renvoient les mêmes listes à l'exécution.

| Tâche | `network-type` | Bibliothèque parser | Familles |
|---|---|---|---|
| Détection | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Classification | 1 | Aucune nécessaire | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentation sémantique | 2 | Aucune nécessaire | pidnet, eomt, dinov2, lingbotvision |
| Segmentation d'instances | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | Aucune nécessaire | yolo9, yolonas, rfdetr, ec |
| Profondeur | 100 | Aucune nécessaire | depth_anything, zipdepth |
| Restauration | 100 | Aucune nécessaire | nafnet, realesrgan, swinir |
| Matting | 100 | Aucune nécessaire | birefnet |
| Regard | 100 | Aucune nécessaire | l2cs |

`network-type=100` signifie que DeepStream n'a pas de post-processeur pour la
tâche. Ces configs mettent `output-tensor-meta=1`, les sorties natives du graphe
passent telles quelles, et l'application les décode depuis les métadonnées de
tenseurs. Les graphes à sorties multiples ne posent pas de problème ici : chaque
couche de sortie arrive dans les métadonnées avec les mêmes noms de sortie et les
mêmes axes dynamiques qu'un export ONNX ordinaire.

Les lignes de segmentation d'instances sont la ligne de détection suivie du
masque de cette instance, aplati à `(netH / 4, netW / 4)`, la résolution que le
parser seg code en dur, sous forme de probabilités pour `segmentation-threshold`.

La classification et le regard tournent en inférence secondaire. Définissez
`process-mode=2` et `operate-on-gie-id` dans la config générée pour placer un
classifieur derrière un détecteur. Le regard est un contrat tête seule, un crop
de visage par entrée : il lui faut donc un détecteur de visages en amont.

Trois familles sont absentes à dessein. `segformer` n'est pas raccordé au contrat
d'export sémantique commun et ne peut s'exporter en ONNX sous aucun format.
RTMDet-Ins et YOLO9 ont leur export de segmentation d'instances bloqué dans
LibreYOLO même. `depth_anything3` n'a pas d'implémentation d'export.

Deux lignes du tableau cachent des trous côté checkpoints. Seul le checkpoint
sémantique EoMT `l` est publié, et la classification DINOv2 n'a aucun checkpoint
publié : cette combinaison exige donc vos propres poids fine-tunés.

## Différences de prétraitement

`nvinfer` calcule `net-scale-factor * (x - offsets)` par canal avec une échelle
scalaire, ce qui ne permet pas d'exprimer un écart-type par canal. Les familles
qui en ont besoin (`rfdetr`, `ec`, les tailles `deimv2` à backbone DINO,
`rtmdet`, `picodet`, et toutes les familles de classification) ont la
normalisation intégrée au graphe exporté, et la config générée alimente le graphe
avec l'espace d'entrée brut correspondant.

C'est sur la géométrie que les pipelines Python de LibreYOLO et `nvinfer`
divergent encore :

- Les familles à letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`,
  `yolo3`, `yolo4`, `yolo7`) remplissent en gris nativement. `nvinfer` remplit en
  noir.
- La détection `yolonas` redimensionne nativement le plus grand côté à 636 dans
  son canevas de 640. Le `maintain-aspect-ratio` de `nvinfer` utilise les 640
  complets.
- La classification redimensionne nativement le plus petit côté puis recadre au
  centre. `nvinfer` étire la frame ou la ROI de l'objet jusqu'à l'entrée réseau,
  donc les sujets recadrés serré diffèrent.
- EoMT exécute nativement des tuiles à fenêtre glissante pour la segmentation
  sémantique. Le graphe exporté est un unique canevas étiré, plus rapide et moins
  exact.
- `pidnet` émet une carte de classes au 1/8 de la résolution d'entrée et
  `lingbotvision` au 1/16. DeepStream suréchantillonne la carte de classes pour
  l'affichage.

Le contrôle de parité ONNX fournit des tenseurs déjà prétraités : il vérifie donc
les sorties du graphe et ne peut pas détecter un mauvais ordre des canaux de
couleur ou une mauvaise politique de padding dans la config. Validez sur vos
propres données avant de déployer une charge de travail à parité exacte.

## Pièges connus

### Deux modèles de détection dans un même répertoire chargent l'engine l'un de l'autre

Chaque config de détection porte la même ligne :

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Le constructeur d'engine du parser impose ce nom de base et il ne varie pas selon
le modèle. Exportez un deuxième modèle de détection dans le même répertoire et le
deuxième lancement charge l'engine mis en cache du premier modèle. Rien ne plante ;
les boîtes sont simplement fausses. Donnez à chaque modèle de détection son propre
répertoire. La validation a dû isoler D-FINE dans un répertoire à lui avant de
pouvoir seulement le tester.

### Une boîte ne peut porter qu'une seule classe

Le format de ligne de `nvinfer` est `[x1, y1, x2, y2, score, class_id]`, une
classe par boîte, donc l'export réduit les scores de classe à leur argmax. Une
boîte que `predict` rapporte sous deux classes survit sous une seule. Cas mesuré :
LibreYOLO rapporte `vase 0.773` et `bottle 0.383` sur la même boîte, et le graphe
DeepStream conserve `vase`. Cela découle du format de ligne du parser et ne peut
pas changer sans sortir de ce contrat : c'est donc un comportement attendu et non
une régression.

## Validé

`deepstream-app` est allé jusqu'à l'EOS avec `App run successful` sur les deux
types de têtes de détection, sur le `sample_1080p_h264.mp4` fourni par NVIDIA
(1443 frames), avec les dumps KITTI par frame activés.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Type de tête | grille | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frames avec détections | 1443 | 1443 |
| Total des détections | 18031 | 71105 |

Les histogrammes de classes sur les 1443 frames placent les voitures en premier
et les personnes en second pour les deux modèles, ce qui est correct pour une
scène de rue. L'écart d'un facteur quatre sur le nombre de détections, c'est la
différence de `cluster-mode` qui fait son travail : D-FINE à `cluster-mode=4` ne
fait aucun clustering, donc chaque requête au-dessus du seuil survit,
quasi-doublons compris.

Deux modèles entraînés indépendamment placent l'objet dominant au même endroit :

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Ce lancement établit cinq choses : TensorRT construit un engine à partir de
l'ONNX exporté sur sm_120, `nvinfer` accepte chaque clé de la config générée,
`NvDsInferParseYolo` lit correctement la disposition des tenseurs, les boîtes
atterrissent en coordonnées 1920x1080 à la résolution source, et les étiquettes
se résolvent contre le fichier d'étiquettes généré.

L'environnement dans lequel il a tourné :

| Composant | Valeur |
|---|---|
| OS hôte | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 Go |
| Pilote | 591.86 |
| Capacité de calcul | 12.0 (Blackwell, sm_120) |
| Runtime de conteneurs | Docker Desktop 29.4.3, backend WSL2 |
| Image DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Version de DeepStream | 8.0.0 |
| CUDA du conteneur | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` à HEAD |

À côté du lancement du pipeline, `tests/unit/test_deepstream_export.py` couvre
les adaptateurs de graphe et les clés de la config générée, et ses 35 tests
passent sur ce commit.

## Non validé

Précisé pour que la portée ci-dessus ne soit pas lue plus largement qu'elle ne
l'est.

- Jetson et aarch64. Le contrat d'export ne dépend pas de l'architecture, mais le
  pipeline n'a été lancé que sur un GPU discret x86.
- Quarante et une des 43 combinaisons. Seules la détection avec `yolo9` et la
  détection avec `dfine` sont passées par DeepStream. La classification, la
  segmentation sémantique, la segmentation d'instances et les tâches à tenseurs
  bruts sont couvertes par des tests unitaires et des contrôles de parité ONNX,
  pas par un lancement de pipeline.
- FP16 et INT8. Seul `network-mode=0` a été exercé.
- Multi-flux et batching. Une seule source, `batch-size=1`.
- L'exactitude face à un dataset de vérité terrain. Les détections ont été
  contrôlées pour leur plausibilité sémantique et leur accord entre modèles, pas
  notées en mAP à travers DeepStream.
