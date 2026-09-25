---
title: "Depth Anything V2 : lancer l'estimation de profondeur avec LibreYOLO"
description: "Depth Anything V2 produit des cartes de profondeur monoculaires parmi les meilleures. Voici comment le lancer en deux lignes avec LibreYOLO."
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Quelle est la façon la plus simple d'utiliser Depth Anything V2 ?"
    a: "Deux lignes avec LibreYOLO : chargez LibreDepthAnythingV2l-depth.pt par son nom, puis appelez predict avec save=True. Les poids sont téléchargés automatiquement à la première utilisation, et la normalisation, la palette de couleurs et le placement sur l'appareil sont gérés pour vous sur CUDA, Apple Silicon ou un CPU standard."
  - q: "Puis-je utiliser Depth Anything V2 à des fins commerciales ?"
    a: "Seul l'encodeur Small est sous licence Apache-2.0. Base, Large et Giant sont sous licence CC-BY-NC-4.0, donc les checkpoints les plus performants sont réservés à un usage non commercial. La licence amont s'applique quelle que soit la bibliothèque qui charge les poids."
  - q: "LibreYOLO prend-il en charge l'entraînement de Depth Anything V2 ?"
    a: "Non. L'entraînement reste dans le dépôt d'origine ; LibreYOLO couvre l'inférence, la vidéo et la validation pour la tâche de profondeur."
---

![Le musée Guggenheim de Bilbao à côté de sa carte de profondeur](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 produit actuellement certaines des meilleures cartes de profondeur monoculaires disponibles.

Si vous connaissez l'écosystème YOLO, le dépôt officiel n'est pas le plus simple à prendre en main. Pour obtenir une seule carte de profondeur, il faut effectuer plusieurs étapes manuelles :

* télécharger le bon checkpoint manuellement et le placer dans le bon dossier,

* faire correspondre l'encodeur à sa configuration (`encoder="vitl"`, `features=256`, `out_channels=...`),

* écrire vous-même l'étape de normalisation et de colorisation,

* gérer le placement sur l'appareil pour que l'exécution fonctionne sur un Mac ou un CPU, et pas seulement sur CUDA,

* et apprendre une API d'inférence qui ne ressemble en rien au reste de votre pile.

Rien de tout cela n'est difficile. Ce sont simplement des frictions que vous pouvez éviter.

LibreYOLO charge les mêmes poids Depth Anything V2 et vous donne accès à la tâche de profondeur avec un seul appel à `predict()`. Indiquez le nom du modèle et son téléchargement se lance à la première utilisation :

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Voilà, c'est tout. Si vous avez déjà utilisé l'API standard de YOLO, la forme vous est familière : chargez un modèle par son nom, appelez `predict` et laissez `save=True` enregistrer la visualisation. C'est exactement le même appel que pour la détection d'objets, sauf que le résultat contient une carte de profondeur plutôt que des boîtes. La palette de couleurs, la normalisation et le placement sur l'appareil sont gérés pour vous. L'exécution se fait de la même façon sous Linux avec CUDA, sur un Mac avec Apple Silicon ou sur un CPU standard. Aucun changement de code.

Le même appel vous donne ensuite accès aux valeurs de profondeur brutes, que vous pouvez exploiter directement. L'entraînement de Depth Anything V2 lui-même reste dans le dépôt d'origine ; LibreYOLO couvre l'inférence, la vidéo et la validation.

Le même appel en une ligne, sur des scènes très différentes :

![L'image d'un parcours de parkour à côté de sa carte de profondeur : les sauteurs au premier plan se détachent des murs en béton à l'arrière-plan.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Vue aérienne de la baie de La Concha à Donostia à côté de sa carte de profondeur : les bateaux et le rivage apparaissent proches, tandis que la pleine mer semble s'éloigner.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![La cour à colonnades de la Casa de Juntas de Gernika à côté de sa carte de profondeur, qui montre l'architecture en perspective.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Une foule lors d'un festival nocturne sur la Plaza de la Constitución à Donostia, à côté de sa carte de profondeur : les rangées de personnes au premier plan se détachent de la façade éclairée à l'arrière-plan.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

À propos des poids : LibreYOLO héberge des checkpoints Depth Anything V2 convertis et les télécharge à la première utilisation. Vous n'avez donc rien à télécharger manuellement. La licence amont s'applique toujours : l'encodeur Small est sous licence Apache-2.0, tandis que Base, Large et Giant sont sous licence CC-BY-NC-4.0 (usage non commercial), les checkpoints les plus performants sont donc réservés à un usage non commercial. Vous préférez travailler hors ligne ou effectuer votre propre conversion ? Le script de conversion à exécuter une fois est toujours disponible :

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Essayer

```bash
pip install libreyolo
```

LibreYOLO est la bibliothèque de vision par ordinateur la plus complète que vous puissiez installer avec pip. Une API familière couvre un catalogue croissant de modèles de pointe : détection d'objets (RF-DETR, D-FINE, DEIM), segmentation, pose, boîtes orientées, et maintenant estimation de profondeur monoculaire avec Depth Anything V2, ainsi que l'entraînement et la validation pour les tâches prises en charge. Elle fonctionne sur tous les principaux systèmes d'exploitation, sur GPU ou CPU, et est entièrement sous licence MIT : vous pouvez donc l'utiliser dans vos produits commerciaux sans aucune contrepartie.

Ajoutez une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
