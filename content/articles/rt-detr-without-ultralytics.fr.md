---
title: "RT-DETR sans Ultralytics : utiliser RT-DETR, v2 et v4 sous Apache-2.0"
description: "Exécutez et affinez RT-DETR, RT-DETRv2 et RT-DETRv4 avec des poids Apache-2.0 dans une bibliothèque MIT : une API pour predict, train, val et export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR est-il gratuit pour un usage commercial ?"
    a: "Oui, si vous utilisez une implémentation permissive. Le code d'origine de RT-DETR et RT-DETRv2 (lyuwenyu/RT-DETR), ainsi que celui de RT-DETRv4 (RT-DETRs/RT-DETRv4), sont sous Apache-2.0, et LibreYOLO exécute les trois versions avec des poids Apache-2.0 dans une bibliothèque MIT. L'implémentation de RT-DETR dans le package ultralytics est sous AGPL-3.0, avec une licence Enterprise payante comme solution de remplacement pour un usage en code fermé. Ces informations sont générales et ne constituent pas un conseil juridique."
  - q: "Quelle est la licence de RT-DETRv4 ?"
    a: "Apache-2.0. Le fichier LICENSE de github.com/RT-DETRs/RT-DETRv4 indique Apache-2.0. RT-DETRv4 utilise un modèle enseignant DINOv3 uniquement pendant l'entraînement ; les poids de l'étudiant publiés ne contiennent aucun paramètre DINOv3, la licence DINOv3 de Meta ne s'applique donc pas à ces poids."
  - q: "Puis-je faire du fine-tuning de RT-DETR sans Ultralytics ?"
    a: "Oui. LibreYOLO fait le fine-tuning des checkpoints de détection RT-DETR, RT-DETRv2 et RT-DETRv4 avec model.train(), ou avec libreyolo train en ligne de commande, en partant des poids COCO publiés. Les modèles à boîtes orientées de RT-DETRv2 sont réservés à l'inférence."
  - q: "Quels poids RT-DETR LibreYOLO fournit-il ?"
    a: "RT-DETR en r18, r34, r50, r50m, r101, l et x ; RT-DETRv2 en r18, r34, r50, r50m et r101 ; RT-DETRv4 en s, m, l et x, tous pour la détection COCO à 640 px. RT-DETRv2 propose aussi les modèles à boîtes orientées n, s, m, l et x, entraînés sur DOTA v1.0 à 1024 px. Chaque fichier est sous Apache-2.0."
---

RT-DETR est un transformeur de détection en temps réel de Baidu. Il décode un ensemble fixe de requêtes au lieu d'une grille dense, ce qui évite d'avoir à régler une étape NMS. Lorsqu'on le recherche, l'une des premières implémentations trouvées est celle du package Python `ultralytics`. Cela soulève une question de licence que le modèle lui-même n'a jamais posée.

## La licence de RT-DETR dépend du dépôt

La licence s'applique à un dépôt, pas à un modèle. RT-DETR est présent dans plusieurs dépôts, dont les licences diffèrent :

| Dépôt | Contenu | Licence |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR et RT-DETRv2, PyTorch et Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR dans le package `ultralytics` | AGPL-3.0 ou licence Enterprise |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 et RT-DETRv4 | Code MIT, poids Apache-2.0 |

L'implémentation d'Ultralytics est de qualité. Sa [page RT-DETR](https://docs.ultralytics.com/models/rtdetr/) répertorie les modèles pré-entraînés `rtdetr-l.pt` et `rtdetr-x.pt`, avec l'entraînement, la validation, l'inférence et l'export. Ils sont fournis dans un package sous licence AGPL-3.0, et Ultralytics vend une licence Enterprise aux équipes qui ne souhaitent pas publier en open source l'ensemble de leur projet. Si aucune option ne convient, l'architecture est disponible sous Apache-2.0 auprès de ses propres auteurs. L'[explication des licences YOLO](/articles/yolo-licenses-explained) décrit le même principe pour la famille YOLO, et le [guide des licences commerciales](/articles/yolo-commercial-license) explique ce que l'AGPL-3.0 implique pour un produit en code fermé.

## Exécuter RT-DETR avec LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Le même appel fonctionne en ligne de commande :

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` et `max_det` filtrent un décodage top-k sur les requêtes et les classes. `iou` est accepté, mais n'est pas utilisé, puisqu'il n'y a pas de NMS. L'objet `Results` renvoyé est celui que retourne chaque modèle LibreYOLO ; remplacer le détecteur ne demande donc qu'une ligne de code.

## RT-DETR, RT-DETRv2 et RT-DETRv4 dans un seul loader

La version figure dans le nom de fichier, et `LibreYOLO()` choisit l'implémentation à partir du checkpoint. Les trois versions se chargent donc de la même façon :

* **RT-DETR :** `LibreRTDETR` en r18, r34, r50, r50m, r101 (backbones ResNet) et l, x (HGNetv2).
* **RT-DETRv2 :** `LibreRTDETRv2` en r18, r34, r50, r50m et r101. Son architecture reprend celle de la version 1, mais modifie l'échantillonnage de l'attention déformable.
* **RT-DETRv4 :** `LibreRTDETRv4` en s, m, l et x. Il réutilise l'architecture de D-FINE, et ses poids proviennent de la distillation d'un modèle enseignant DINOv3 vers un modèle étudiant HGNetv2.

Tous les poids de détection sont entraînés sur COCO et fonctionnent à 640 px. À titre de référence, les README amont indiquent 46.5 AP pour RT-DETR-R18 et 53.0 AP pour RT-DETR-L sur COCO, et de 49.8 AP pour RT-DETRv4-S jusqu'à 57.0 AP pour RT-DETRv4-X. La [page de documentation RT-DETR](/docs/models/rt-detr) contient le tableau de benchmark LibreYOLO pour chaque checkpoint.

RT-DETRv2 prend aussi en charge les boîtes orientées. `LibreRTDETRv2n-obb.pt` à `LibreRTDETRv2x-obb.pt` sont les checkpoints officiels DOTA v1.0, avec 15 classes aériennes à 1024 px, convertis au format LibreYOLO :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

La [détection à boîtes orientées](/docs/tasks/oriented-detection) décrit le format des étiquettes et les métriques.

## Faire du fine-tuning de RT-DETR sur vos données

L'entraînement démarre à partir d'un checkpoint publié :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Pour un vrai entraînement, indiquez le YAML de votre propre dataset dans `data`. Chaque version a son propre learning rate par défaut, et les versions 1 et 2 fixent le learning rate du backbone au vingtième de `lr0`, conformément à la recette d'origine. Le multi-GPU se configure avec `device=0,1` dans l'interface CLI, et le fine-tuning par adaptateurs [LoRA](/docs/train/lora) avec `lora=True`, une fois l'extra `lora` installé. Les modèles à boîtes orientées de RT-DETRv2 sont réservés à l'inférence. Consultez la page [entraînement](/docs/train) pour les datasets, l'augmentation et les loggers.

## Valider et exporter

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` renvoie un dictionnaire simple. Les formats d'export comprennent ONNX, TorchScript, ExecuTorch, TensorRT et OpenVINO ; la matrice d'export de la [page du modèle](/docs/models/rt-detr) indique ceux qui sont validés pour chaque tâche. Un fichier `.onnx` ou `.engine` exporté se charge à nouveau avec `LibreYOLO()` et renvoie le même objet `Results`. Consultez les guides par format dans la page [export](/docs/export).

## Quand choisir les dépôts d'origine

Si vous devez reproduire un résultat publié avec les configurations exactes des auteurs, utiliser la version Paddle ou réaliser vous-même la distillation du modèle enseignant DINOv3 de RT-DETRv4, utilisez les dépôts amont. LibreYOLO fait le fine-tuning de l'étudiant v4 publié ; il n'exécute pas le modèle enseignant. Et si votre projet est déjà sous AGPL-3.0 ou couvert par une licence Enterprise d'Ultralytics, il n'y a aucune raison liée aux licences de changer.

## Remarque sur la licence

Le code de LibreYOLO est sous MIT. Chaque fichier de poids RT-DETR est sous Apache-2.0, et chaque dépôt de poids Hugging Face fournit la licence LICENSE et le fichier NOTICE de la version amont, que l'Apache-2.0 demande de conserver lorsque vous redistribuez les poids. Les poids que vous entraînez sur vos propres données vous appartiennent. RT-DETRv4 utilise DINOv3 uniquement comme modèle enseignant pendant l'entraînement, et les étudiants publiés ne contiennent aucun paramètre DINOv3. Ces informations sont générales et ne constituent pas un conseil juridique. Vérifiez les fichiers LICENSE actuels avant toute distribution.

## Essayer

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO est sous licence MIT, fonctionne sous Linux, Mac et Windows, et s'exécute sur GPU, Apple Silicon et CPU sans changer de code.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation RT-DETR](/docs/models/rt-detr)
