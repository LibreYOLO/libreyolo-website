---
title: Installer
seo_title: Installer LibreYOLO
description: >-
  Installer LibreYOLO depuis PyPI, choisir les extras facultatifs nécessaires à
  une famille de modèles ou à une cible d'exportation, et vérifier que PyTorch
  détecte votre GPU.
lead: >-
  LibreYOLO est publié sur PyPI sous le nom libreyolo. Le paquet de base couvre
  la prédiction, l'entraînement, la validation et les familles de modèles qui
  n'ont besoin de rien d'autre que PyTorch. Les extras facultatifs ajoutent le
  reste.
keywords:
  - installer libreyolo
  - pip install libreyolo
  - extras libreyolo
  - cuda libreyolo
  - gpu libreyolo
  - dépendances libreyolo
last_verified: 1.5.0
meta:
  - label: Paquet
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 ou version ultérieure
  - label: Licence du code
    value: MIT
  - label: Dépendance principale
    value: PyTorch 2.4 ou version ultérieure
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Avec des extras
      language: bash
      code: >
        # Séparez les extras par des virgules pour les combiner dans une
        installation.

        pip install "libreyolo[rfdetr,onnx]"
    - label: Tout installer
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Depuis les sources
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, chaque GPU visible et les paquets
        # facultatifs installés.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inventaire des modèles
      language: bash
      code: |
        # Chaque famille enregistrée avec ses tâches, ses tailles et ses
        # résolutions d'entrée. Les familles dont l'extra manque sont
        # accompagnées de la commande pip qui les active.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Installer

<code-tabs name="install" />

Python 3.10 ou une version ultérieure est requis. L'installation de base
récupère PyTorch, torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss,
tqdm, pycocotools, typer, click, safetensors et SciPy. YOLOv9 et les autres
familles qui ne demandent rien de plus fonctionnent donc immédiatement après
`pip install libreyolo`.

Un clone récupère `release`, la branche stable dont le code correspond à cette
documentation. La branche d'intégration, qui contient les travaux non publiés,
est `dev`.

## Extras facultatifs

Un extra est un nom entre crochets qui ajoute les dépendances nécessaires à une
famille de modèles ou à une cible d'exportation. Rien d'autre ne change : l'API
reste identique, que l'extra soit présent ou non.

### Familles de modèles

| Extra | Ajouts |
|---|---|
| `rfdetr` | `transformers`, qui fournit le backbone de RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, qui fournit les encodeurs ViT-L/16 et EfficientNet-Lite3 de MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` et `bitsandbytes` hors macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` et `ftfy`, nécessaires au tokenizer de texte CLIP intégré |
| `siglip2` | `sentencepiece`, nécessaire au tokenizer multilingue SigLIP 2 |
| `gaze` | `gdown`, qui active le téléchargement automatique du checkpoint L2CS |
| `rtdetr` | Rien. RT-DETR n'a besoin d'aucune dépendance supplémentaire. Le nom reste stable |

### Exportation et environnements d'exécution

| Extra | Ajouts |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 et `pycuda`, hors macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, macOS uniquement |
| `tflite`, alias `litert` | `libreyolo[onnx]` avec `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` et `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` avec `MNN` |
| `ncnn` | `pnnx` et `ncnn` |
| `paddle` | `libreyolo[onnx]` avec `paddlepaddle` 2.6.2 et `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` pour l'inférence V2 par HTTP et HTTPS |

### Entraînement, évaluation et journalisation

| Extra | Ajouts |
|---|---|
| `lora` | `libreyolo[rfdetr]` avec `peft`, pour le fine-tuning avec `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, le backend C++ d'évaluation COCO |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` est facultatif plutôt qu'une dépendance obligatoire, afin qu'une
plateforme dépourvue de wheel précompilée ne puisse pas empêcher une
installation simple. En l'absence du paquet, l'évaluation COCO revient à
pycocotools et l'exécution se poursuit.

### Outils

| Extra | Ajouts |
|---|---|
| `stream` | `yt-dlp`, nécessaire uniquement pour résoudre les URL de pages YouTube |
| `tracking` | Rien. Toutes les dépendances de suivi sont déjà des dépendances principales |
| `label` | `libreyolo[sam]`, qui active l'aide clic-vers-masque dans `libreyolo label` |
| `hub-kernels` | `kernels`, le chargeur facultatif de kernels Hub compilés. Consultez la page sur les [kernels](/docs/reference/kernels), qui précise que son installation peut modifier les prédictions RF-DETR dans les limites de tolérance des nombres flottants |
| `clip-convert` | `libreyolo[clip]` avec `open_clip_torch`, pour la conversion des poids et les vérifications de parité |
| `siglip2-convert` | `libreyolo[siglip2]` avec `transformers`, pour la même raison |

Les webcams, RTSP, RTMP, TCP, UDP, HLS et les listes locales de plusieurs flux
ne nécessitent aucun extra. Seules les URL de pages YouTube en ont besoin.

### Extra global

`libreyolo[all]` installe en une seule commande les extras de modèles,
d'exportation, de suivi et de journalisation. Certains en sont volontairement
exclus. `neptune` ne figure pas dans cet ensemble, car la version stable de
`neptune-scale` exige une version de protobuf antérieure à 7, tandis que le
parcours TFLite exige protobuf 7. `executorch` est exclu parce qu'ExecuTorch
impose les versions de PyTorch avec lesquelles il fonctionne, et `coreai` parce
que `coreai-torch` fixe PyTorch à la série 2.11.x et ferait migrer tout
l'environnement vers cette version. `fast-eval`, `hub-kernels`,
`clip-convert` et `siglip2-convert` sont également laissés de côté. Installez
chacun d'eux par son nom.

## Contraintes des plateformes

Trois extras sont limités à certaines plateformes par les marqueurs de leurs
dépendances. L'installation réussit donc partout et installe simplement moins
de composants lorsqu'aucune wheel n'existe.

| Extra | Contrainte |
|---|---|
| `coreai` | macOS uniquement. La chaîne d'outils Core AI ne convertit ni ne s'exécute ailleurs |
| `tensorrt` | Ignoré sur macOS, qui ne dispose pas de CUDA |
| `tflite`, `litert` | `onnx2tf` et `ai-edge-litert` nécessitent Python 3.12 ou une version ultérieure |

`sensenova` ignore `bitsandbytes` sur macOS, où aucune wheel n'est publiée. Le
reste de l'extra s'installe normalement.

Si l'espace disque est la contrainte, PyTorch en occupe la majeure partie, et
l'essentiel de PyTorch provient de la charge utile CUDA incluse dans sa wheel
par défaut. Une wheel limitée au CPU la supprime sans rien sacrifier. Pour
exécuter une détection ONNX sur une machine qui ne doit contenir aucune
installation de torch, consultez l'[installation légère](/docs/lightweight-install).

## GPU et CUDA

Le périphérique est sélectionné lors de la construction du modèle. La valeur
par défaut, `device="auto"`, utilise CUDA lorsque
`torch.cuda.is_available()` vaut true, puis Metal Performance Shaders lorsque
`torch.backends.mps.is_available()` vaut true, et le CPU dans les autres cas.
Rien d'autre dans la bibliothèque n'inspecte le matériel. Si PyTorch ne détecte
pas un GPU, LibreYOLO ne le peut pas non plus.

Pour imposer le périphérique, transmettez `device` au modèle ou à `predict`,
`train`, `val` et `export`. Il accepte `"cpu"`, `"cuda"`, `"cuda:0"`,
`"mps"`, un entier seul tel que `0` ou une chaîne de chiffres telle que `"0"`.
Les deux dernières formes sont développées en `cuda:<n>`.

Commencez par `libreyolo checks`. Cette commande affiche la version de Torch,
les versions de CUDA et de cuDNN avec lesquelles Torch a été construit, ainsi
que chaque GPU visible et sa mémoire. Si elle n'indique aucun CUDA sur une
machine équipée d'une carte NVIDIA, la wheel PyTorch choisie par pip est une
version CPU. Installez d'abord une version CUDA depuis l'index PyTorch, puis
LibreYOLO :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Il s'agit du même index que celui fixé par le dépôt pour son propre
environnement géré par uv sous Linux et Windows. Il nécessite le pilote NVIDIA
555 ou une version ultérieure, conformément aux exigences de l'environnement
d'exécution CUDA 12.8. macOS conserve la wheel PyPI, car l'hôte de téléchargement
PyTorch ne publie aucune version Darwin.

## Vérifier l'installation

<code-tabs name="verify" />

`libreyolo models` est le moyen le plus rapide de vérifier qu'un extra a pris
effet : une famille dont la dépendance manque est affichée avec la commande pip
exacte qui l'active. Les deux commandes acceptent également `--json`, qui
affiche les mêmes données sur stdout sous forme d'objet lisible par une machine.
