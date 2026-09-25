---
title: "RTMDet sans mmdetection : exécuter le modèle sans compiler mmcv"
description: "RTMDet est l'un des détecteurs les plus rapides et exacts. L'installation officielle échoue avec toute version de PyTorch publiée au cours des deux dernières années. Voici l'alternative."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Pourquoi mmcv ne s'installe-t-il pas avec les versions récentes de PyTorch ?"
    a: "La version la plus récente de mmcv, la 2.2.0 d'avril 2024, ne fournit des binaires précompilés que jusqu'à torch 2.4 / CUDA 12.1. Avec toute version plus récente de PyTorch, mmcv doit compiler ses opérations C++/CUDA depuis les sources. Cela prend 10 à 30 minutes et nécessite un toolkit CUDA correspondant, nvcc et un compilateur C++ compatible. C'est de là que viennent des erreurs comme l'absence du module mmcv._ext."
  - q: "Puis-je exécuter RTMDet sans installer mmdetection ?"
    a: "Oui. LibreYOLO charge les poids RTMDet convertis à partir des checkpoints OpenMMLab en amont, et l'inférence est bit à bit équivalente à celle de mmdetection avec le même checkpoint. Chargez LibreRTMDets.pt par son nom pour déclencher son téléchargement automatique. Les cinq tailles, de Tiny à X, fonctionnent toutes en 640 px."
  - q: "RTMDet est-il gratuit pour un usage commercial ?"
    a: "Oui. MMDetection et RTMDet sont sous Apache 2.0, les poids convertis conservent les mêmes conditions Apache 2.0, et le code de LibreYOLO est sous MIT. Aucune restriction commerciale."
  - q: "Dans quels cas mmdetection reste-t-il le bon choix ?"
    a: "Si vous devez entraîner RTMDet ou faire du fine-tuning avec l'ensemble du pipeline d'augmentation de MMDetection, ou si vous êtes déjà très investi dans l'écosystème OpenMMLab et que mmcv fonctionne. Pour l'inférence et le déploiement, LibreYOLO est la meilleure option."
---

RTMDet est un détecteur en temps réel d'OpenMMLab qui atteint une bonne exactitude sur COCO tout en restant assez rapide pour le déploiement. Son architecture est claire. Son installation ne l'est pas.

Pour exécuter RTMDet à partir de la source officielle, il faut assembler une pile OpenMMLab à quatre couches :

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Les contraintes de version sont strictes. mmdet 3.3.0 exige `mmcv < 2.2.0`, mais `mim install mmcv>=2.0.0` installe sans problème la version 2.2.0, qui échoue ensuite sur une assertion à l'exécution. Vous devez fixer la version manuellement.

Puis vient le problème principal. La version la plus récente de mmcv est la 2.2.0, publiée en avril 2024 et jamais mise à jour depuis. Elle ne fournit des binaires précompilés que jusqu'à **torch 2.4 / CUDA 12.1**. Aujourd'hui, un `pip install torch` tout neuf vous donne PyTorch 2.12. Cet écart oblige mmcv à compiler ses opérations C++/CUDA depuis les sources : 10 à 30 minutes, un toolkit CUDA correspondant, `nvcc` et un compilateur C++ compatible. C'est là que s'accumulent les erreurs documentées :

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- les opérations compilées n'ont pas été construites ou ne correspondent pas,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` sur toute carte RTX de la série 30,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- la pile vous oblige à revenir à Python 3.8,

* un segmentation fault à l'importation dû à une incompatibilité de version de GCC.

La FAQ officielle d'OpenMMLab conseille d'installer mmcv avec pip plutôt qu'avec mim pour éviter le piège des versions. Sa page Get Started conseille d'utiliser mim. Les deux documents sont officiels. Ils se contredisent.

Une fois la pile installée, l'inférence exige encore de choisir un fichier de configuration dont le nom encode la recette d'entraînement, puis de télécharger séparément un checkpoint au nom contenant son hash et de le relier via un registre qu'il faut d'abord apprendre à utiliser.

LibreYOLO remplace tout cela :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Pas de `mim`, pas de matrice de versions à quatre paquets, pas de compilation depuis les sources, pas de fichiers de configuration, pas de recherche d'un checkpoint par son hash, pas de contrainte à Python 3.8. Les poids sont convertis à partir des checkpoints OpenMMLab en amont, et l'inférence est bit à bit équivalente à celle de mmdetection avec le même checkpoint.

Cinq tailles sont disponibles : Tiny, Small, Medium, Large et X. Elles fonctionnent toutes en 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Dans quels cas choisir encore la solution d'origine

Si vous devez entraîner RTMDet ou faire du fine-tuning avec l'ensemble du pipeline d'augmentation de MMDetection, ou si vous êtes déjà très investi dans l'écosystème OpenMMLab et que mmcv fonctionne, restez-y. Pour l'inférence et le déploiement, LibreYOLO est la meilleure option.

## À propos de la licence

MMDetection et RTMDet sont sous Apache 2.0. Le code de LibreYOLO est sous MIT. Les poids conservent les mêmes conditions Apache 2.0 que celles de la source amont. Aucune restriction commerciale.

## Essayer

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO est sous licence MIT, fonctionne sous Linux, Mac et Windows, et tourne sur GPU, Apple Silicon ou CPU sans changer de code. Une seule API couvre RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, la segmentation, l'estimation de pose, la profondeur et plus encore.

Ajoutez une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
