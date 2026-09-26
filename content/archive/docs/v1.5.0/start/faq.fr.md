---
title: FAQ
seo_title: FAQ LibreYOLO
description: "Réponses courtes aux questions qui concernent tous les modèles LibreYOLO\_: matériel, licences, poids, appareils, entraînement, couverture de l'export et CLI."
lead: >-
  Réponses aux questions qui ne concernent pas une famille de modèles précise.
  Les informations propres à une famille figurent sur la page de celle-ci.
keywords:
  - faq libreyolo
  - gpu requis libreyolo
  - licence libreyolo
  - emplacement poids libreyolo
  - cli libreyolo
  - libreyolo hors ligne
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Par quel modèle commencer\u00a0?

YOLOv9 pour un détecteur CNN et RF-DETR pour un transformer. Tous deux
appartiennent au niveau flagship, ce qui signifie que les fonctionnalités sont
conçues et validées sur GPU avec eux avant tout autre modèle. Consultez
[YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr) ou
[tous les modèles](/docs/models).

## Un GPU est-il nécessaire\u00a0?

Non. Chaque modèle s'exécute sur CPU et tout le
[démarrage rapide](/docs/quickstart) est conçu pour y fonctionner. Un GPU
modifie la durée de l'entraînement et de l'inférence vidéo, pas leur
fonctionnement.

## Comment LibreYOLO choisit-il un appareil\u00a0?

La valeur par défaut est `device="auto"`, qui utilise CUDA lorsque PyTorch
l'indique disponible, puis Metal Performance Shaders s'il est disponible, et
sinon CPU. Pour fixer le choix, transmettez `device` au modèle ou à `predict`,
`train`, `val` et `export`. Il accepte `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`,
un entier seul comme `0` ou une chaîne de chiffres. Les deux dernières formes
deviennent `cuda:<n>`.

`libreyolo checks` affiche le build Torch, ses versions CUDA et cuDNN et chaque
GPU visible. Si cette commande n'affiche aucun CUDA, la wheel PyTorch est un
build CPU. La page [Installation](/docs/install) explique comment la remplacer.

## Où sont enregistrés les poids téléchargés\u00a0?

Dans `weights/` par rapport au répertoire de travail. Une référence de modèle
sans répertoire y est résolue et téléchargée à la première utilisation. Une
référence qui contient un répertoire est utilisée exactement telle quelle et
n'est jamais récupérée. Consultez les
[checkpoints et poids](/docs/weights).

## Puis-je travailler sans accès réseau\u00a0?

Oui. Récupérez les checkpoints une fois sur une machine connectée, copiez le
répertoire `weights/` et plus aucun accès réseau n'aura lieu. Un chemin partagé
en lecture seule fonctionne aussi, puisqu'une référence contenant un
répertoire est interprétée littéralement. Les datasets sont résolus sous
`~/datasets` ou sous `LIBREYOLO_DATASETS_DIR`.

## Puis-je utiliser LibreYOLO à des fins commerciales\u00a0?

Le code est sous licence MIT. Les poids pré-entraînés sont une question
distincte\u00a0: ils peuvent hériter des conditions du projet ou du dataset dont
ils proviennent, et ces conditions ne sont pas uniformes, même au sein d'une
famille. La licence du dépôt Hugging Face précis fait autorité, et chaque page
de modèle contient une section Licence qui la reproduit. Lorsque les poids
sont restreints, LibreYOLO affiche la restriction avant le début du
téléchargement.

## Puis-je charger un checkpoint d'un autre projet\u00a0?

Généralement, en transmettant son chemin à `LibreYOLO()`. Les structures
upstream reconnues sont converties lors du chargement en conservant le nombre
et les noms de classes, puis un checkpoint LibreYOLO est écrit à côté de la
source. La page [Importer des poids existants](/docs/migrate) décrit les formats
reconnus et ceux qui nécessitent un script de conversion.

## Pourquoi train lève-t-il NotImplementedError\u00a0?

Parce que cette famille est réservée à l'inférence et l'exception en donne la
raison. La prédiction, la validation et, lorsque cette fonctionnalité est prise
en charge, l'export fonctionnent tous. LibreYOLO ne possède simplement aucune
boucle d'entraînement pour cette architecture. Le niveau de prise en charge
dans l'en-tête d'une page de modèle vous l'indique avant l'essai. Consultez les
[concepts fondamentaux](/docs/concepts).

## Que renvoie val\u00a0?

Un dictionnaire simple, et non un objet. Les clés de détection comprennent
`metrics/precision`, `metrics/recall`, `metrics/mAP50` et
`metrics/mAP50-95`. Les autres tâches renvoient les clés pertinentes, comme
`metrics/accuracy_top1` pour la classification ou `metrics/PQ`, `metrics/SQ`
et `metrics/RQ` pour la segmentation panoptique.

## Comment exécuter une prédiction sur un dossier, une vidéo ou une webcam\u00a0?

Transmettez l'élément comme source. Un chemin de fichier représente une image,
un répertoire toutes les images qu'il contient, un chemin vidéo une vidéo, un
entier un indice de webcam, et une URL RTSP, RTMP, TCP, UDP ou HLS un flux en
direct. Un fichier `.streams` énumère plusieurs sources. Les sources en direct
exigent `stream=True`, qui produit un objet `Results` par image au lieu de
construire une liste. Ce paramètre est aussi utile pour les longues vidéos et
les grands répertoires. Seules les URL de pages YouTube nécessitent un extra,
`libreyolo[stream]`.

## Comment ne conserver que certaines classes\u00a0?

Transmettez à `predict` les indices de classes voulus dans `classes`, par
exemple `classes=[0, 2]`. `conf` définit le seuil de confiance, 0.25 par
défaut, et `max_det` limite les détections par image, 300 par défaut.

## Le CLI utilise-t-il des options ou des paires clé=valeur\u00a0?

Une clé et une valeur reliées par un signe égal, pour chaque commande\u00a0:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` accepte un chemin ou un nom court de la forme `family-size`, avec un
suffixe de tâche facultatif. `libreyolo models` énumère toutes les valeurs
valides. Les commandes de diagnostic et d'inventaire acceptent aussi `--json`,
qui affiche les mêmes données comme objet lisible par une machine sur stdout.

## Tous les modèles s'exportent-ils vers tous les formats\u00a0?

Non. La couverture dépend de la famille et de la tâche, et chaque format
possède son propre extra à installer. Chaque page de modèle contient la matrice
d'export de sa famille. La [section Export](/docs/export) présente les formats.

## Quelle différence entre segment, semantic et panoptic\u00a0?

Ce sont trois tâches distinctes. `segment` produit un masque par objet détecté.
`semantic` attribue une classe à chaque pixel sans distinguer les instances.
`panoptic` attribue exactement une étiquette à chaque pixel et réunit les
éléments dénombrables avec les régions amorphes. Elles possèdent des vérités
terrain, champs de résultats et métriques différents. Une famille prend en
charge celles qui figurent dans sa liste de tâches.

## Comment entraîner sur mes propres classes\u00a0?

Écrivez un YAML de dataset avec `train`, `val` et `names`. Les étiquettes sont
placées à côté des images dans une arborescence `labels/` parallèle, un fichier
`.txt` par image, avec des coordonnées normalisées. `nc` est facultatif et doit
correspondre à `names` s'il est présent. Exécutez d'abord
`libreyolo doctor <data.yaml>`\u00a0: il recherche les problèmes du dataset et se
termine avec un code différent de zéro s'il trouve des erreurs, ce qui permet
de l'utiliser comme contrôle CI.

## Pourquoi le chargement affiche-t-il un avertissement de métadonnées\u00a0?

Parce que le checkpoint ne contient pas toutes les métadonnées v1.0. Le
chargement continue par un chemin de compatibilité et l'avertissement nomme
exactement les clés manquantes. Exécutez
`libreyolo metadata path=<file>` pour afficher le contenu et consultez les
[checkpoints et poids](/docs/weights) pour connaître les exigences du schéma.

## Un import ne fonctionne plus après une mise à niveau. Qu'est-ce qui a changé\u00a0?

Deux classes ont été renommées par cohérence\u00a0: `LibreYOLORTDETR` est devenue
`LibreRTDETR` et `LibreYOLORFDETR` est devenue `LibreRFDETR`. Les anciens noms
se résolvent toujours et émettent un `DeprecationWarning` qui indique le
nouveau. Le code existant continue ainsi de fonctionner pendant sa mise à jour.

