---
title: Installation légère
seo_title: Exécuter l'inférence ONNX de LibreYOLO sans PyTorch
description: >-
  Installer LibreYOLO avec --no-deps et exécuter la détection ONNX uniquement
  avec numpy, sans torch sur le disque. La méthode, ses limites et la liste
  exacte des paquets.
lead: >-
  Le parcours d'inférence ONNX de LibreYOLO utilise numpy de bout en bout, y
  compris pour le décodage et la NMS. Il n'a aucun besoin de PyTorch à
  l'exécution. Une installation qui ignore la résolution des dépendances peut
  donc effectuer la détection sans que torch soit présent sur la machine.
keywords:
  - inférence sans torch
  - libreyolo sans pytorch
  - inférence onnx sans torch
  - installation légère libreyolo
  - pip install no-deps
  - réduire espace disque libreyolo
  - inférence onnxruntime
last_verified: 1.5.0
meta:
  - label: S'applique à
    value: 'Détection ONNX, sept familles de modèles'
  - label: Point d'entrée
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Niveau de prise en charge
    value: 'Au mieux, pas une distribution distincte'
snippets:
  install:
    - label: Installation légère
      language: bash
      code: |
        # Installez le paquet sans sa liste de dépendances, puis ajoutez les
        # quatre paquets réellement importés par le parcours de détection ONNX.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: torch limité au CPU
      language: bash
      code: |
        # Essayez d'abord cette option. Elle conserve toutes les fonctionnalités
        # et évite la wheel CUDA, qui occupe l'essentiel de l'espace disque.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # Ici, xyxy est un ndarray numpy et non un tenseur torch.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Pourquoi cela fonctionne

`pip install --no-deps libreyolo` installe le paquet tout en ignorant
entièrement sa liste de dépendances. Rien n'est résolu pour vous, et il vous
revient d'installer ce que vous utilisez réellement.

Cette approche n'est utile que si le parcours de code souhaité ne dépend
réellement pas des composants ignorés. C'est bien le cas pour la détection ONNX.
Le décodage, y compris la suppression des non-maxima, utilise numpy. Les
recettes de prétraitement aussi. PyTorch est une dépendance pour l'entraînement
et l'inférence eager, mais n'est jamais appelé sur ce parcours.

Avant cette version, l'importation échouait malgré tout : importer quoi que ce
soit sous `libreyolo.models` construisait toutes les classes de modèles pour
alimenter le registre de détection automatique des checkpoints, et ces classes
sont des sous-classes de `torch.nn.Module`. Les recettes de prétraitement
résident désormais dans leur propre paquet, `libreyolo.preprocess`, et
l'importation de torch est différée jusqu'à l'accès à un attribut torch. Le
parcours ONNX peut ainsi être importé sans que torch soit installé sur la
machine. Ce paquet contient un préprocesseur natif numpy par famille :
`yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`, `dfine`, `deim` et
`deimv2`, soit deux de plus que les sept familles vérifiées de bout en bout
ci-dessous. Chaque fichier `libreyolo/models/<family>/utils.py` le réexporte,
ce qui préserve les chemins d'importation existants.

## Essayer d'abord la wheel limitée au CPU

La plupart des personnes qui recherchent cette solution veulent éviter une
installation de plusieurs gigaoctets. Cette taille est concentrée à un seul
endroit : la wheel `torch` par défaut inclut CUDA. Une version limitée au CPU
est bien plus petite et ne demande aucun parcours d'installation particulier.

<code-tabs name="install" />

L'option limitée au CPU conserve toutes les fonctionnalités de LibreYOLO :
entraînement, validation, toutes les tâches, toutes les familles et la CLI.
Choisissez l'installation légère lorsque vous ne voulez aucun torch sur la
machine, pas seulement une version moins volumineuse.

## Ce que couvre l'installation légère

| | |
|---|---|
| Tâche | Détection |
| Format | ONNX |
| Point d'entrée | `OnnxBackend` |
| Interface | Bibliothèque Python |

Sept familles ont été vérifiées sur ce parcours :
[YOLOv9](/docs/models/yolov9), [YOLO-NAS](/docs/models/yolo-nas),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) et
[DEIM](/docs/models/deim), en comptant les variantes de chaque famille.

Il s'agit du périmètre vérifié, pas d'une limite imposée par la bibliothèque.
Les autres tâches et familles sortent simplement du champ testé : certaines
chargeront torch lorsque vous les appellerez, et quelques-unes pourront
fonctionner. Considérez tout ce qui dépasse cette liste comme non testé, et non
comme pris en charge ou défectueux.

Dans ce périmètre, les résultats sont identiques à ceux de l'installation
normale, et pas seulement proches. Chaque famille a été exportée vers ONNX et
exécutée deux fois, une fois normalement et une fois avec torch bloqué. Les
boîtes, scores et classes correspondaient exactement. Un test de parité dans la
suite empêche ce contrat de dériver.

## Les cinq pièges courants

**Utilisez `OnnxBackend`, pas les classes de modèles.**
`LibreYOLO9("model.onnx")` nécessite toujours torch, car `LibreYOLO9` est
elle-même une sous-classe de `nn.Module`. C'est l'erreur la plus probable,
puisque toutes les autres pages de cette documentation chargent un modèle par
sa classe ou par `LibreYOLO()`.

**Exportez ailleurs.** La production du fichier `.onnx` nécessite torch. La
machine légère ne peut donc pas le créer. Exportez-le sur une machine de
développement ou de CI, puis livrez l'artefact à la cible allégée.

**Les résultats contiennent des tableaux numpy.** Ici,
`result.boxes.xyxy` est un `ndarray`. Les conteneurs acceptent les deux types,
donc les noms d'attributs ne changent pas, mais un code qui appelle `.cpu()` ou
`.numpy()` sur un résultat échouera.

**Une seule image renvoie un seul `Results`.** `predict()` renvoie un
`Results` pour une image et une liste pour plusieurs. Indexer un résultat
unique avec `[0]` sélectionne la première détection, pas la première image. Vous
obtenez alors silencieusement un résultat à une boîte au lieu d'une erreur.

**La CLI ne fonctionnera pas.** `typer` et `click` ne font pas partie des
quatre paquets. La commande `libreyolo` n'est donc pas disponible. Il s'agit
d'une installation de bibliothèque.

## Prédire

<code-tabs name="predict" />

Remplacez `onnxruntime` par `onnxruntime-gpu` pour utiliser CUDA. Ces quatre
paquets sont ceux qu'un appel complet à `predict()` sans torch importe
réellement. Ils ont été relevés pendant l'appel, et non déduits par
raisonnement. `opencv-python-headless` remplace le paquet déclaré
`opencv-python` : même module, aucune bibliothèque d'interface graphique et
moins d'espace disque.

Parmi les autres dépendances déclarées, `requests` n'est nécessaire que pour
charger une image depuis une URL, `pycocotools` et `scipy` servent à la
validation et à l'évaluation, tandis que `typer` et `click` appartiennent à la
CLI.

## Cette liste dérivera volontairement

La liste de paquets ci-dessus est exacte pour la version indiquée en haut de
cette page. `--no-deps` vous soustrait à la résolution des dépendances. Rien ne
la contrôle donc pour vous, et une version ultérieure peut importer un élément
qui ne figure pas ici.

Si vous rencontrez une `ModuleNotFoundError`, vous connaissez déjà la méthode :
installez le paquet manquant. C'est le modèle de maintenance prévu, et non un
rapport de bogue. Ce parcours est proposé au mieux et ne constitue pas une
distribution prise en charge séparément. C'est aussi pourquoi il n'existe aucun
second paquet léger sur PyPI et qu'aucun n'est prévu.

Pour confirmer que votre environnement est réellement dépourvu de torch, au
lieu de revenir discrètement à une installation existante, vérifiez-le :

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Il est utile de conserver cette vérification dans la CI de l'image allégée.
Sans elle, un environnement qui possède torch par hasard réussira tous les
tests sans rien vous apprendre.
