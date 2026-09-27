---
title: NVIDIA Jetson
seo_title: Installer LibreYOLO et PyTorch sur NVIDIA Jetson
description: "Installer LibreYOLO sur un NVIDIA Jetson\_: les quatre bibliothèques CUDA que JetPack laisse de côté, l'étape --no-deps dont PyTorch a besoin, et des chiffres mesurés sur Orin Nano."
lead: >-
  Les cartes NVIDIA Jetson font tourner LibreYOLO avec les wheels PyTorch
  aarch64 standard. Aucun build torch spécifique à Jetson n'intervient, mais
  JetPack omet quatre bibliothèques auxquelles torch se lie, et l'installation
  doit les fournir.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - installer pytorch sur jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - tensorrt sur jetson
  - wheels aarch64
last_verified: 1.4.0
meta:
  - label: Carte
    value: "Jetson Orin Nano Super Developer Kit, 8\_Go, compute capability GPU 8.7"
  - label: Plateforme
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Stack testée
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, le 2026-07-27
  - label: Absent de JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Benchmarks
    value: >-
      223 exécutions vérifiées sur cette carte, 58 modèles répartis sur 12
      familles, en PyTorch, ONNX Runtime et TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Suivi dans
    value: La moitié Jetson de l'issue 648
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Recette d'installation et sortie attendue reprises de l'installation du
  2026-07-27 sur un Jetson Orin Nano Super. Les lignes de latence et
  d'exactitude viennent de l'instantané de résultats vérifiés derrière
  visionanalysis.org, filtré sur le matériel jetson_orin, mesuré en juin 2026
  sur libreyolo 1.2.0.dev0. Comportement de l'export et du chargeur lu dans
  libreyolo/export/exporter.py, libreyolo/export/tensorrt.py et
  libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Paquets système et environnement virtuel
      language: bash
      code: |
        # JetPack n'installe ni pip ni le module venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: 'PyTorch, depuis l''index de wheels CUDA 13'
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: Les quatre bibliothèques que JetPack ne fournit pas
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Si pip exige cuda-toolkit 13.0.3, installez avec --no-deps'
      language: bash
      code: |
        # Avec --no-deps, les dépendances de torch sont nommées à la main.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Nommer la prochaine bibliothèque manquante au lieu de deviner
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Tout ce qui manque encore dans les bibliothèques de torch, en une
        passe :

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Installer LibreYOLO après torch, pas avant'
      language: bash
      code: |
        # torch est déjà satisfait, donc pip laisse en place le build CUDA.
        pip install libreyolo

        # L'extra ONNX ne sert qu'à l'export. Un export TensorRT passe par
        # ONNX, ajoutez-le donc avant la section export ci-dessous.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versions et périphérique
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Puis exécuter un vrai kernel
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Télécharge le checkpoint à la première utilisation.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Écrit libreyolo9s.onnx, puis en construit libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Le moteur se recharge par le même point d'entrée.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Mode d'alimentation et fréquences
      language: bash
      code: >
        sudo nvpmodel -q      # les modes exposés par cette carte, et celui
        actif

        sudo nvpmodel -m 0    # mode maximal sur la carte testée ici

        sudo jetson_clocks


        tegrastats            # charge en direct ; nvidia-smi est limité sur
        Tegra
source_hash: c07ff908503e89b5
---

## Ce que consigne cette page

Cette page consigne une configuration vérifiée de bout en bout, pas une matrice
de compatibilité. La carte était un Jetson Orin Nano Super Developer Kit doté de
8 Go de mémoire, sous JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13,
Python 3.12.3), et la stack qui a démarré dessus était `libreyolo 1.4.0` avec
`torch 2.13.0+cu130`, OpenCV 5.0.0 et NumPy 2.5.1.
`torch.cuda.is_available()` a renvoyé `True` et le GPU s'est identifié comme
`Orin`.

Les autres versions de JetPack, les autres cartes Jetson et les autres versions
de CUDA n'ont pas été testées. La recette ci-dessous est celle qui a fonctionné
sur cette combinaison.

Cette exécution date du 2026-07-27 et portait sur LibreYOLO 1.4.0 ; elle n'a pas
été refaite sur du matériel en 1.5.0 : c'est la seule page de l'arbre 1.5.0 à
porter encore une vérification 1.4.0, d'où le `last_verified: "1.4.0"` de son
front matter. Rien dans les changements de la 1.5.0 ne touche au chemin
d'installation, aux quatre bibliothèques manquantes ni aux flags d'export
décrits ici, les commandes devraient donc rester valables, mais les numéros de
version dans les sorties ci-dessous sont ceux qu'a affichés la 1.4.0, pas une
mesure en 1.5.0.

Deux points vont à l'encontre de ce que disent la plupart des guides Jetson. Les
wheels sont les builds aarch64 ordinaires publiés pour CUDA 13, aucun build
torch spécifique à Jetson n'est donc nécessaire. Et JetPack ne fournit pas
quatre bibliothèques auxquelles ces wheels se lient, si bien qu'`import torch`
échoue une bibliothèque à la fois jusqu'à ce que les quatre soient installées.

## Installation

Les images JetPack arrivent sans pip et sans le module `venv` : les deux passent
donc en premier.

<code-tabs name="prep" />

Une carte de 8 Go est juste pour les checkpoints les plus gros. Ajouter du swap
sur le NVMe avant de les charger évite un kill pour manque de mémoire en pleine
exécution.

Ensuite PyTorch. L'index CUDA 13 porte les wheels aarch64 ; l'index
supplémentaire fournit les dépendances pur Python depuis PyPI.

<code-tabs name="torch" />

Les quatre wheels `nvidia-*-cu13` sont la partie facile à oublier. JetPack
fournit le pilote GPU, pas cuDNN, NCCL, cuSPARSELt ni NVSHMEM, et torch refuse
de s'importer sans elles. Installer les quatre d'un coup est plus rapide que de
les découvrir une exception à la fois.

Le troisième extrait couvre un échec précis : les métadonnées de dépendances de
torch pour le build CUDA 13 réclament `cuda-toolkit==13.0.3`, qui n'a pas de
wheel aarch64 sur PyPI, et la résolution échoue donc avant le moindre
téléchargement. `--no-deps` court-circuite le résolveur, ce qui oblige à nommer
chaque dépendance sur la ligne de commande.

LibreYOLO s'installe en dernier. L'installer en premier laisse pip choisir son
propre torch, qui sur cette plateforme n'est pas le build CUDA.

<code-tabs name="install" />

Toutes les dépendances restantes se résolvent en wheels aarch64 précompilées, y
compris OpenCV, NumPy, SciPy, pycocotools et safetensors. Rien ne se compile
depuis les sources.

## Vérifier que CUDA fonctionne

<code-tabs name="verify" />

Le deuxième extrait compte autant que le premier. Une wheel compilée pour la
mauvaise architecture GPU annonce quand même `torch.cuda.is_available() == True`,
puis échoue à la première opération réelle avec `CUDA error: no kernel image is
available for execution on the device`. Une multiplication de matrices sur le
périphérique est le contrôle qui l'attrape.

## Lancer une prédiction

<code-tabs name="predict" />

`predict` renvoie le même objet `Results` que sur n'importe quelle autre
plateforme, les pages des modèles s'appliquent donc sans changement.

## Exporter vers TensorRT

Sur cette carte, TensorRT a été plus rapide que PyTorch et que ONNX Runtime pour
les 55 modèles mesurés dans tous les runtimes.

<code-tabs name="export" />

`format="tensorrt"` écrit d'abord un graphe ONNX et construit le moteur à partir
de lui, l'extra `onnx` doit donc être installé. `LibreYOLO()` s'oriente d'après
le suffixe du fichier : un fichier `.engine` se charge par le même appel qu'un
checkpoint `.pt`.

N'utilisez pas l'extra pip `tensorrt` sur un Jetson. Il épingle `tensorrt-cu12`,
un build CUDA 12, face à une plateforme CUDA 13. Utilisez plutôt le TensorRT
qu'installe JetPack. Si `import tensorrt` échoue dans l'environnement virtuel
alors qu'il fonctionne à l'extérieur, recréez l'environnement avec
`--system-site-packages` pour que le module système soit visible.

Les moteurs TensorRT sérialisés sont liés à l'appareil, à l'architecture GPU et
à la version de TensorRT qui les a construits. Un moteur construit sur une
station de travail ne se chargera pas sur un Jetson : l'étape de construction
s'exécute donc sur la carte.

## Mesures sur cette carte

Latence par image, taille de batch 1, de bout en bout, prétraitement et
post-traitement compris, sur COCO val2017 (sous-ensemble de 500 images) avec
`conf=0.001` et `max_det=300`. Cinq modèles sur les 58 mesurés :

| Modèle | Entrée (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

La colonne mAP est le score propre de l'exécution TensorRT FP16. Sur les
55 modèles mesurés dans les quatre runtimes, le plus grand écart entre le score
PyTorch FP32 et le score TensorRT FP16 était de 0.59 point, sur DEIMv2-X. Les
runtimes diffèrent en vitesse, pas en exactitude.

TensorRT FP32 a été plus rapide que PyTorch et que ONNX Runtime pour ces
55 modèles. TensorRT FP16 a lui aussi été plus rapide que PyTorch FP32 sur les
55, d'un facteur 1.68x à 6.22x, avec une médiane de 3.39x. ONNX Runtime est
celui qui varie : il a été plus lent que PyTorch sur 23 des 55, dont la ligne
RT-DETR-r18.

Conditions derrière chaque chiffre : `libreyolo 1.2.0.dev0`,
`torch 2.12.0+cu130`, Python 3.12.3, CUDA 13, pilote 595.78, ONNX Runtime
1.24.0, mesuré en juin 2026. La latence sur un Jetson dépend aussi du mode
d'alimentation actif, que les enregistrements de benchmark ne portent pas.

<code-tabs name="power" />

Les 223 exécutions, y compris les 53 autres modèles et les colonnes
d'exactitude complètes, sont publiées sur
[la page Jetson Orin de Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Dépannage

### import torch échoue en nommant une bibliothèque partagée

L'une des quatre bibliothèques ci-dessus manque. Plutôt que de deviner
laquelle, lisez-la depuis le binaire :

<code-tabs name="ldd" />

Chaque entrée manquante correspond à une wheel :

| Bibliothèque manquante | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch avertit qu'aucun build ne prend en charge ce GPU

Le premier appel CUDA sur la configuration qui fonctionne affiche ceci :

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

L'avertissement est cosmétique sur cette carte. La wheel porte des kernels
`sm_80` et l'Orin les exécute. Le même avertissement apparaissait sur la wheel
précédente de cet index, celle qui a produit chaque ligne de benchmark
ci-dessus. Confirmez avec la multiplication de matrices de la vérification CUDA
plutôt que de faire confiance ou non au message.

### CUDA error: no kernel image is available for execution on the device

La wheel installée a été compilée pour une autre architecture GPU. C'est ce qui
arrive avec les wheels de l'index `sbsa` de NVIDIA, qui visent les GPU ARM
serveur plutôt que le silicium Jetson. Réinstallez depuis l'index CUDA 13 de la
section installation.

### pip ne trouve pas cuda-toolkit 13.0.3

Il n'existe pas de wheel aarch64 pour lui. Utilisez la forme `--no-deps` de la
section installation et nommez explicitement les dépendances de torch.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

La wheel torch aarch64 se lie aux NVIDIA Performance Libraries pour le calcul
CPU. Installez-les et placez-les sur le chemin des bibliothèques :

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Cet index convient pour ces deux bibliothèques CPU. Ce sont ses builds torch qui
produisent l'échec « no kernel image » ci-dessus.

### Sources de wheels incompatibles avec JetPack 7.2

| Source | Résultat sur l'Orin Nano Super |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130` torch | Compilé pour les GPU ARM serveur. S'importe, annonce CUDA disponible, puis échoue avec « no kernel image is available for execution on the device ». |
| `pypi.jetson-ai-lab.io/jp6/*` torch | Builds CUDA 12 et Python 3.10. Ils ne s'installent pas sur le Python 3.12 de cette image. |
| Conteneurs PyTorch JetPack 6 | L'initialisation CUDA échoue avec l'erreur 801 sur un hôte JetPack 7. |
| Compiler torch depuis les sources | Fonctionne, mais prend des heures sur une carte de 8 Go et devient inutile une fois les wheels CUDA 13 installées. |

## DeepStream

Pour un pipeline vidéo complet plutôt qu'une boucle Python, exportez avec
`deepstream=True` et faites passer le graphe par `nvinfer`. Ce chemin a sa
propre page, avec la config `nvinfer` générée, la compilation du parseur de
bounding boxes et les pièges connus : [DeepStream](/docs/export/deepstream).

Le pipeline DeepStream lui-même a été validé sur un GPU dédié x86, pas sur un
Jetson. Le contrat d'export ne dépend pas de l'architecture, mais l'exécution du
pipeline sur aarch64 reste à faire.

## Non vérifié

- Les versions de JetPack autres que 7.2, et les versions de L4T autres que
  R39.2.
- Les cartes Jetson autres que l'Orin Nano Super 8 Go.
- L'entraînement sur la carte. L'inférence et l'export ont été exercés ; pas un
  entraînement.
- Les moteurs INT8. Seules des lignes FP32 et FP16 existent pour cette carte.
- Les tailles de batch supérieures à 1. Toutes les mesures ci-dessus sont en
  batch 1.
