---
title: Fine-tuning LoRA
seo_title: Fine-tuning LoRA dans LibreYOLO
description: >-
  Affinez un détecteur transformer avec peu de VRAM grâce à lora=True. Les neuf
  familles prises en charge, la recette d'adaptateur de chacune et le
  comportement des checkpoints.
lead: >-
  LoRA gèle les parties lourdes pré-entraînées d'un modèle et entraîne à côté de
  petits adaptateurs de faible rang, ainsi que les couches qui doivent rester
  denses. Dans LibreYOLO, toute l'interface publique tient en un booléen.
keywords:
  - fine-tuning lora
  - fine-tuning efficace en paramètres
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
    - label: L'export fusionne les adaptateurs
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

Sans elle, `lora=True` lève une `ImportError` qui nomme cette commande au lieu
de lancer accidentellement un fine-tuning complet.

## Utiliser LoRA

<code-tabs name="train" />

`lora=True` constitue toute l'interface. Le rang, alpha, le dropout et les
modules cibles sont fixés pour chaque famille afin de correspondre à sa
référence en amont, et ne sont pas des paramètres exposés aux utilisateurs.

Une famille qui ne prend pas LoRA en charge provoque une erreur pendant la
configuration au lieu d'ignorer le flag :

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

La CLI le rejette plus tôt, avant la construction du modèle, au moyen de sa
propre liste d'autorisation des neuf mêmes familles.

## Familles prises en charge

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 et v4, EC et ConvNeXt. La barrière
est l'attribut `supports_lora` de la classe du trainer de chaque famille, et la
CLI contient une liste d'autorisation correspondante.

La couverture des tâches est plus étroite que celle des familles. D-FINE et EC
prennent uniquement la détection en charge, et leurs chemins de segmentation et
de pose provoquent une erreur. Le chemin sémantique de RF-DETR provoque une
erreur. ConvNeXt est destiné à la classification.

Tout le reste provoque une erreur. Il n'existe aucun mode partiel ou silencieux.

## Fonctionnement de chaque recette

Les recettes diffèrent parce que les architectures diffèrent, et une recette
qui fonctionne sur un backbone ViT n'a aucun point d'attache sur un backbone
convolutionnel.

RF-DETR utilise DoRA, la version de LoRA qui décompose les poids, avec un rang
16 et un alpha 16 sur les projections d'attention `query`, `key` et `value` du
backbone DINOv2, conformément à la référence RF-DETR. Le backbone ViT est gelé ;
le projecteur, le décodeur et la tête de détection continuent de s'entraîner
normalement.

D-FINE, DEIM et RT-DETR v1, v2 et v4 associent un backbone convolutionnel à un
encodeur hybride transformer et à un décodeur déformable, si bien que la
séparation se déplace. Le backbone convolutionnel est entièrement gelé, ce qui
ignore aussi sa passe backward. Les blocs transformer gèlent leurs poids de
base et entraînent des adaptateurs LoRA simples au même rang 16 et alpha 16 sur
leurs couches linéaires : les couches feed-forward `linear1` et `linear2`, la
gate et les projections de l'attention déformable. Tout le reste, soit la
fusion convolutionnelle de l'encodeur, les projections d'entrée, les têtes de
prédiction et les embeddings de requêtes, continue de s'entraîner de façon
dense.

Deux détails de cette recette sont volontaires. L'auto-attention du décodeur
reste gelée sans adaptateurs, car `nn.MultiheadAttention` de PyTorch lit
directement `out_proj.weight` et contournerait silencieusement un adaptateur
injecté. Il s'agit aussi de LoRA simple plutôt que de DoRA, car plusieurs
couches linéaires du décodeur sont initialisées à zéro par conception, et la
normalisation de magnitude de DoRA divise par la norme des poids.

DEIMv2 reprend la même recette avec ses couches feed-forward SwiGLU `w12` et
`w3` comme cibles. Ses tailles S, M, L et X possèdent également un backbone ViT
DINOv3, dont la base ViT est gelée et les couches d'attention fusionnées `qkv`
reçoivent des adaptateurs, tandis que la pyramide convolutionnelle Spatial
Tuning Adapter continue de s'entraîner comme l'équivalent du projecteur. Ces
adaptateurs `qkv` sont insérés même lorsque la configuration fournit le ViT
gelé, puisque l'adaptation d'un backbone gelé est précisément l'objectif. Les
tailles inférieures à S utilisent un backbone convolutionnel et suivent la
recette simple.

EC est un DETR dont le backbone est un ViT entouré d'une pyramide de projection
convolutionnelle entraînable. La base ViT est gelée et ses couches `qkv`
reçoivent des adaptateurs, les blocs transformer suivent la recette partagée,
et le projecteur ainsi que les têtes restent denses.

Les blocs ConvNeXt contiennent des MLP linéaires channels-last, `fc1` et `fc2`,
qui reçoivent des adaptateurs simples. Les convolutions depthwise, les normes et
les paramètres layer-scale sont gelés. La tête de classification reste dense,
si bien que les nombres de classes personnalisés continuent de fonctionner.

Les têtes de détection et de classification restent toujours entraînables dans
toutes les recettes, car un nombre de classes personnalisé exige une tête
nouvellement entraînée.

## Checkpoints et export

`best.pt` et `last.pt` conservent les tenseurs des adaptateurs. Une exécution
LoRA se reprend ou s'inspecte donc comme n'importe quelle autre. Le chargement
de l'un de ces checkpoints nécessite l'installation de l'extra `lora`, car le
chargeur rejoue l'injection des adaptateurs pour aligner les clés.

`export()` fusionne les adaptateurs dans les poids denses, si bien qu'un
artefact exporté ne dépend pas de `peft`. La même fusion est directement
disponible pour un modèle en mémoire.

<code-tabs name="merge" />

Après une fusion, l'arborescence de modules est entièrement dense et une
seconde fusion ne fait rien.

## Économies et limites

LoRA réduit la mémoire de l'optimiseur et des gradients et, pour les familles
qui gèlent entièrement leur backbone, évite aussi la passe backward de ce
backbone.

La mémoire des activations reste inchangée. Les activations forward doivent
toujours être conservées pour tout ce qui reste entraînable, et ce sont
généralement elles qui fixent le pic. Pour le budget VRAM le plus serré,
réduisez aussi `batch` ou `imgsz`.

## Pages connexes

- [Gel des couches](/docs/train/layer-freezing) pour l'autre façon d'entraîner
  une partie des poids, qui fonctionne sur chaque famille sans dépendance
  supplémentaire. `freeze` et `lora=True` se combinent : les paramètres des
  adaptateurs restent entraînables même lorsque leur groupe de backbone parent
  est gelé.
- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `imgsz` et le
  reste de `train()`.
