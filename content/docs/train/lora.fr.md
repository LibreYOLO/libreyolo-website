---
title: Fine-tuning LoRA
seo_title: Fine-tuning LoRA dans LibreYOLO
description: >-
  Affiner un détecteur Transformer avec peu de VRAM grâce à lora=True. Les neuf
  familles compatibles, la recette d'adaptateurs de chacune et le comportement
  des checkpoints.
lead: >-
  LoRA gèle les parties lourdes pré-entraînées d'un modèle et entraîne à côté de
  petits adaptateurs de faible rang, ainsi que les couches qui doivent rester
  denses. Dans LibreYOLO, toute l'interface publique tient en un booléen.
keywords:
  - fine tuning lora
  - fine tuning économe en paramètres
  - peft
  - dora
  - entraînement faible vram
  - rf-detr lora
  - d-fine lora
  - fusion adaptateurs
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: L'exportation fusionne les adaptateurs
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Fusionner sur place
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Installer

LoRA repose sur la dépendance facultative `peft`.

<code-tabs name="install" />

En son absence, `lora=True` déclenche une `ImportError` qui indique cette
commande, au lieu d'effectuer accidentellement un fine-tuning complet.

## L'utiliser

<code-tabs name="train" />

`lora=True` constitue toute l'interface. Le rang, alpha, le dropout et les
modules cibles sont fixés par famille conformément à chaque référence amont et
ne sont pas des réglages exposés à l'utilisateur.

Une famille qui ne prend pas LoRA en charge déclenche une erreur à la
configuration plutôt que d'ignorer l'option :

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

La CLI le refuse plus tôt encore, avant la construction du modèle, à l'aide de
sa propre liste des neuf mêmes familles autorisées.

## Familles compatibles

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 et v4, EC et ConvNeXt. Le contrôle
repose sur l'attribut `supports_lora` de la classe du programme d'entraînement
de chaque famille. La CLI contient une liste correspondante.

La couverture des tâches est plus étroite que celle des familles. D-FINE et EC
prennent uniquement en charge la détection, et leurs parcours de segmentation
et de pose déclenchent une erreur. Le parcours sémantique de RF-DETR déclenche
une erreur. ConvNeXt couvre la classification.

Tout le reste déclenche une erreur. Il n'existe aucun mode partiel ou silencieux.

## Effet de chaque recette

Les recettes diffèrent parce que les architectures diffèrent. Une recette
adaptée à un backbone ViT ne trouve rien sur lequel s'attacher dans un backbone
convolutif.

RF-DETR emploie DoRA, LoRA à décomposition des poids, au rang 16 et avec
alpha 16 sur les projections d'attention `query`, `key` et `value` du backbone
DINOv2, conformément à la référence RF-DETR. Le backbone ViT est gelé. Le
projecteur, le décodeur et la tête de détection continuent de s'entraîner
normalement.

D-FINE, DEIM et RT-DETR v1, v2 et v4 associent un backbone convolutif à un
encodeur Transformer hybride et à un décodeur déformable. La séparation se
déplace donc. Le backbone convolutif est entièrement gelé, ce qui évite aussi
sa rétropropagation. Les blocs Transformer gèlent leurs poids de base et
entraînent des adaptateurs LoRA ordinaires de rang 16 et alpha 16 sur leurs
couches linéaires : `linear1` et `linear2` de la propagation avant, la porte et
les projections de l'attention déformable. Tout le reste, notamment la fusion
convolutive de l'encodeur, les projections d'entrée, les têtes de prédiction et
les embeddings de requêtes, continue de s'entraîner de façon dense.

Deux détails de cette recette sont volontaires. L'auto-attention du décodeur
reste gelée sans adaptateur, car `nn.MultiheadAttention` de PyTorch lit
directement `out_proj.weight` et contournerait silencieusement un adaptateur
injecté. Il s'agit en outre de LoRA ordinaire plutôt que de DoRA, car plusieurs
couches linéaires du décodeur sont initialisées à zéro par conception et la
normalisation de magnitude de DoRA divise par la norme des poids.

DEIMv2 reprend la même recette en ciblant `w12` et `w3` de ses couches de
propagation avant SwiGLU. Ses tailles S, M, L et X possèdent aussi un backbone
ViT DINOv3. La base ViT y est gelée, et ses couches d'attention fusionnées
`qkv` reçoivent des adaptateurs, tandis que la pyramide convolutive Spatial
Tuning Adapter continue de s'entraîner comme l'analogue du projecteur. Ces
adaptateurs `qkv` sont ajoutés même si la configuration publie le ViT gelé,
puisque l'objectif est précisément d'adapter un backbone gelé. Les tailles
inférieures à S ont un backbone convolutif et suivent la recette ordinaire.

EC est un DETR dont le backbone ViT est entouré d'une pyramide de projecteurs
convolutifs entraînable. La base ViT est gelée et ses couches `qkv` reçoivent
des adaptateurs. Les blocs Transformer suivent la recette partagée, tandis que
le projecteur et les têtes restent denses.

Les blocs ConvNeXt possèdent des MLP linéaires à canaux en dernière dimension,
`fc1` et `fc2`, qui reçoivent des adaptateurs ordinaires. Les convolutions
depthwise, les normalisations et les paramètres d'échelle des couches sont
gelés. La tête de classification reste dense afin de conserver la prise en
charge des nombres de classes personnalisés.

Les têtes de détection et de classification restent toujours entraînables dans
chaque recette, car un nombre de classes personnalisé exige une tête entièrement
entraînée.

## Checkpoints et exportation

`best.pt` et `last.pt` conservent les tenseurs des adaptateurs. Une exécution
LoRA reprend donc ou s'inspecte comme toute autre. Le chargement d'un de ces
checkpoints nécessite l'installation de l'extra `lora`, car le chargeur rejoue
l'injection des adaptateurs pour faire correspondre les clés.

`export()` fusionne les adaptateurs dans les poids denses. Un artefact exporté
n'a donc aucune dépendance envers `peft`. La même fusion est directement
disponible sur un modèle en mémoire.

<code-tabs name="merge" />

Après une fusion, l'arborescence des modules est entièrement dense et une
seconde fusion ne fait rien.

## Économies et limites

LoRA réduit la mémoire de l'optimiseur et des gradients. Sur les familles qui
gèlent entièrement leur backbone, il évite également la rétropropagation de ce
backbone.

La mémoire des activations ne change pas. Les activations de propagation
doivent toujours être conservées pour tout ce qui reste entraînable, ce qui
détermine généralement le pic. Pour les budgets de VRAM les plus serrés,
réduisez aussi `batch` ou `imgsz`.

## Voir aussi

- [Gel des couches](/docs/train/layer-freezing) pour l'autre manière
  d'entraîner un sous-ensemble des poids, compatible avec chaque famille et sans
  dépendance supplémentaire. `freeze` et `lora=True` se composent : les
  paramètres des adaptateurs restent entraînables même lorsque leur groupe
  parent est gelé.
- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `imgsz` et le
  reste de `train()`.
