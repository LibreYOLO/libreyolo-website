---
title: Les meilleures alternatives à Ultralytics en 2026
description: "Un guide pratique des meilleures alternatives open source à Ultralytics YOLO en 2026 : licences, tâches couvertes, limites de déploiement et place de LibreYOLO, la bibliothèque YOLO sous licence MIT la plus complète."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLO est-il gratuit pour un usage commercial ?"
    a: "Uniquement sous AGPL-3.0, qui exige de publier le code source de l'application que vous construisez avec, y compris les services accessibles en réseau. Un usage commercial à code fermé nécessite la licence commerciale payante d'Ultralytics. Les alternatives permissives (MIT, Apache-2.0) évitent cette contrainte."
  - q: "Quelle alternative à Ultralytics présente le moins de contraintes de licence ?"
    a: "Pour une stack permissive, déployable partout : LibreYOLO (code MIT), RF-DETR (Apache-2.0) et YOLOX (Apache-2.0). Tous évitent le copyleft de l'AGPL."
  - q: "Qu'est-ce qui remplace YOLO en 2026 ?"
    a: "De plus en plus, les détecteurs à base de transformers en temps réel : RT-DETR, RF-DETR et les familles D-FINE et DEIM. Ils égalent ou dépassent l'exactitude de YOLO à latence comparable sur du matériel suffisamment puissant. Les CNN de type YOLO gardent l'avantage sur les petits appareils edge, où ces transformers fonctionnent mal et manquent d'une solution NCNN ou CPU mature."
  - q: "Ces outils peuvent-ils fonctionner sur un Raspberry Pi ou un NPU ?"
    a: "Cela dépend de l'export. Les détecteurs CNN comme YOLOX et RTMDet s'exportent vers NCNN et fonctionnent sur un Pi, et certains (YOLOX) prennent en charge le NPU Hailo. Ce n'est pas le cas des détecteurs à base de transformers comme RF-DETR ; ils nécessitent un GPU ou un CPU puissant."
---

Transparence : LibreYOLO est notre projet et figure en tête de cette liste. Tous les autres outils présentés ici sont ceux que nous utilisons dans notre travail, et nous indiquons où ils font mieux que LibreYOLO.

La plupart des équipes quittent Ultralytics pour l'une de ces trois raisons :

- **La licence.** Les modèles YOLO d'Ultralytics sont sous AGPL-3.0 (YOLOv5 depuis 2023, et à partir de v8). Le scénario est connu : vous construisez un produit, vous le livrez, puis un examen juridique ou un client soulève la question du modèle, et le choix se résume à ouvrir le code de toute votre application ou à acheter une licence commerciale. Le copyleft de l'AGPL s'étend à votre code dès que vous le distribuez ou le proposez en service. C'est une raison fréquente de chercher ailleurs.
- **L'ouverture.** Un seul fournisseur possède les modèles, le code d'entraînement et les conditions d'utilisation. Certaines équipes veulent des poids et du code sous licences permissives, sur lesquels elles peuvent construire sans demander d'autorisation.
- **Le modèle.** YOLO n'est pas toujours la meilleure architecture pour la tâche. Des détecteurs à base de transformers en temps réel comme RT-DETR, RF-DETR et D-FINE peuvent offrir une meilleure exactitude à latence égale. Le problème est qu'ils sont éparpillés dans des dépôts de recherche et, comme vous le verrez, LibreYOLO exécute les trois avec une seule API. C'est donc une raison de changer de bibliothèque, pas de la quitter pour un seul modèle.

Chaque outil ci-dessous est évalué sur trois points : une licence qui vous laisse livrer votre produit, une installation et une exécution sur un PyTorch actuel, et un export vers le matériel sur lequel vous déployez réellement.

Un constat revient au fil de la lecture : chaque alternative est bonne, mais pénible à utiliser seule. YOLOX s'installe à grand-peine, MMDetection est un purgatoire de versions de wheels verrouillées, la famille RT-DETR est du code de recherche sans assistance, Detectron2 est figé depuis 2021. LibreYOLO est en tête de cette liste parce qu'il intègre déjà la plupart de ces outils, avec une maintenance, sous MIT, sur une stack actuelle et derrière une API familière. C'est moins un élément de cette liste que la couche qui les réunit.

## Comparatif rapide

| Outil | Licence | Tâches couvertes | Idéal pour | Export edge |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (code) | Détection, segmentation, pose, classification, profondeur, regard, suivi | Le point d'accès MIT à plus de 20 familles de modèles avec une seule API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Détection, segmentation, pose (préversion) | Détection et segmentation en production | ONNX, TFLite, TensorRT ; ni NCNN ni Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Pré-entraînement, distillation | Données non étiquetées, distillation de DINOv2/v3 | Sans objet (ce n'est pas un détecteur) |
| **YOLOX** | Apache-2.0 | Détection | Détection en temps réel sans ancres | Installation amont difficile, fonctionne via LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Tout (recherche) | Diversité des architectures, reproduction d'articles scientifiques | Via export |
| **Detectron2** | Apache-2.0 | Détection, segmentation, points clés | Mask R-CNN et la famille R-CNN | Manuel |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Détection | DETR en temps réel à l'état de l'art | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Détection, segmentation, pose | ViT compacts pour l'edge, distillés à partir de DINOv3 | Niveau recherche |
| **RT-DETR (v1-v4)** | Apache-2.0 | Détection | Détection en temps réel de pointe | ONNX, TensorRT |

## 1. LibreYOLO

Licence : MIT (code). Idéal si vous cherchez la bibliothèque YOLO sous licence MIT la plus complète et une alternative directement substituable à Ultralytics.

**LibreYOLO est la bibliothèque YOLO sous licence MIT qui exécute tous les modèles avec une seule API.** C'est notre projet et, à la lecture du reste de cette liste, son rôle devrait être clair : c'est le point d'accès qui fait tourner les alternatives. LibreYOLO intègre déjà presque tous les dépôts pénibles cités plus haut, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM, la famille RT-DETR, derrière une API familière et maintenue. Vous obtenez le modèle sans fouilles archéologiques pour l'installation ni contrainte de licence.

La raison d'être du projet dépasse la commodité. YOLO était au départ une recherche ouverte : chacun pouvait librement utiliser Darknet de Joseph Redmon, de v1 à v3, et construire dessus. L'ère AGPL l'a refermé. LibreYOLO existe pour rendre ce travail à nouveau accessible, comme ses créateurs l'avaient voulu : sous licence permissive, maintenu par la communauté, sans transmission de données à votre insu. Il y a donc deux aspects.

**Premier aspect : c'est l'alternative à Ultralytics YOLO sous licence MIT.** Il conserve l'API que vous connaissez déjà, donc vos datasets au format YOLO et vos scripts se portent avec peu de changements, mais sous MIT au lieu d'AGPL : aucun copyleft qui s'étend à votre code, aucune licence commerciale à acheter, et aucune télémétrie qui transmet des données comme Ultralytics le fait par défaut. Si vous êtes ici pour la licence, c'est tout l'intérêt. YOLO9 et RF-DETR sont les choix par défaut, largement testés et prêts pour la production ; le reste du catalogue est plus récent et clairement indiqué comme expérimental. Nous préférons le dire plutôt que de prétendre que tout a été éprouvé sur le terrain.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # ou LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Deuxième aspect : son périmètre dépasse largement celui d'Ultralytics.** Une seule API couvre plus de 20 familles de modèles, YOLO à base de CNN, DETR à base de transformers, backbones ViT, SAM et d'autres, et non le catalogue d'un seul fournisseur. Et il n'y a pas que des détecteurs :

- **Tâches :** segmentation d'instances et sémantique, pose, classification et boîtes orientées, plus deux tâches totalement absentes d'Ultralytics : la profondeur monoculaire (Depth Anything V2) et l'estimation du regard (L2CS).
- **Les outils pratiques :** suivi multi-objets avec des identifiants persistants (ByteTrack et OC-SORT), inférence vidéo avec sous-échantillonnage des images et aperçu en direct, inférence par tuiles pour les images de drones et satellites, interface dans le navigateur par glisser-déposer (`libreyolo ui`), et commande `doctor` qui vérifie votre dataset avant de gaspiller un entraînement.
- **Un entraînement complet :** accumulation de gradients, gel de couches, reprise, augmentation au moment du test, fine-tuning LoRA/DoRA, multi-GPU et journalisation TensorBoard/MLflow/Weights & Biases.
- **Export :** sept formats (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) avec quantification INT8/FP16, NMS intégré et métadonnées du modèle intégrées.
- **Exécution partout :** accepte les chemins, URL, PIL, NumPy, tenseurs ou octets bruts, et fonctionne sur CUDA, Apple Silicon (MPS) ou un simple CPU sans changer le code.

Quand le dépôt d'un bon détecteur est abandonné ou difficile à installer, LibreYOLO exécute ses poids sur une stack actuelle et maintenue.

## 2. RF-DETR

Licence : Apache-2.0 (Nano, Small, Medium, Large). Idéal pour la détection et la segmentation en production.

RF-DETR, de Roboflow et publié à ICLR 2026, est le modèle de cette liste le plus prêt pour la production. Il est bien conçu, vient d'une équipe qui livre des produits et constitue un excellent premier choix quand vous avez besoin d'une détection ou d'une segmentation qui fonctionne et à laquelle vous pouvez vous fier dans un système réel. Le package `rfdetr` et les poids Nano à Large sont sous Apache-2.0. Les poids plus grands XL et 2XL utilisent la licence PML de Roboflow.

## 3. Lightly

Idéal pour les données non étiquetées, le pré-entraînement auto-supervisé et la distillation d'un backbone DINO.

Lightly n'est pas un détecteur. Il sert à tirer de la valeur de données que vous n'avez pas étiquetées et se compose de deux parties aux licences différentes.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Un framework de composants d'apprentissage auto-supervisé : fonctions de perte, têtes, augmentations, banques de mémoire et implémentations de référence de plus de vingt méthodes (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE et d'autres). Vous écrivez la boucle d'entraînement ; il fournit tout le nécessaire pour pré-entraîner une représentation à partir de zéro sur des images non étiquetées.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (avec des options commerciales et communautaires gratuites).** La solution clé en main, celle à laquelle la plupart des gens font référence. Indiquez-lui un modèle de fondation comme DINOv2 ou DINOv3 et vos données non étiquetées : en quelques lignes, sans étiquettes, il distille ce backbone en un modèle plus petit que vous pouvez déployer (YOLO, RT-DETR, ViT ou un réseau personnalisé). Attention à la licence : AGPL-3.0 est le même copyleft qui pousse à chercher des alternatives à Ultralytics. Lisez donc les conditions si vous comptez livrer un produit à code fermé, ou prenez la licence commerciale.

Choisissez Lightly quand il vous manque un bon backbone et que vous n'avez pas d'étiquettes. Il s'associe aux détecteurs de cette liste plutôt que de leur faire concurrence. Les plus récents le montrent aussi : RT-DETRv4 intègre la distillation d'un modèle de fondation visuel à l'entraînement, et DEIMv2 intègre des backbones DINOv3 directement dans le détecteur.

## 4. YOLOX

Licence : Apache-2.0. Idéal pour la détection en temps réel sans ancres, une fois installé.

Le dépôt amont est figé depuis sa dernière version, v0.3.0 en avril 2022, avec plus de 700 issues ouvertes, et son installation sur une stack de 2026 est pénible. Il n'y a pas de `pyproject.toml`, et `requirements.txt` verrouille `onnx-simplifier==0.4.10`, qui n'a pas de wheels précompilées au-delà de Python 3.10. Sur un interpréteur moderne, il se compile donc depuis les sources et nécessite cmake ainsi qu'une chaîne de compilation C++. La version de NumPy n'est pas verrouillée non plus, ce qui ouvre la porte à des incompatibilités d'ABI entre NumPy 2.0 et d'anciennes versions de pycocotools et torch. La détection fonctionne une fois la chaîne de dépendances résolue, mais y parvenir a un coût.

Le modèle lui-même reste bon : un détecteur en temps réel sans ancres de Megvii, avec code et poids sous Apache-2.0, et une base de comparaison raisonnable même si de nouveaux détecteurs l'ont depuis dépassé. Remettre en état un dépôt abandonné mais bien conçu est désormais le genre de tâche qu'un agent de programmation réalise en un après-midi. L'état de l'installation est donc moins bloquant qu'il n'y paraît. LibreYOLO exécute aussi directement les poids YOLOX sur une stack actuelle. Dans les deux cas, l'architecture vaut la peine d'être utilisée. Nous avons écrit un tutoriel plus complet [ici](/articles/yolox-with-libreyolo).

## 5. L'univers MMDetection

Licence : principalement Apache-2.0. Idéal pour la diversité des architectures et la reproduction d'articles scientifiques.

Pendant des années, MMDetection a hébergé l'implémentation officielle de presque tous les articles sur la détection : Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN, des centaines de configurations. En 2026, OpenMMLab a mis fin progressivement à la série mm. La dernière version de [MMDetection](https://github.com/open-mmlab/mmdetection) était v3.3.0 en janvier 2024, les issues restent sans réponse et la stack est bloquée par le verrouillage de version de `mmcv`, dont les wheels précompilées sont en retard sur les versions actuelles de PyTorch et CUDA. Une installation neuve tend donc à échouer tant que vous n'avez pas rétrogradé l'ensemble. Nous avons décrit ce problème [ici](/articles/rtmdet-without-mmdetection).

Une entreprise néerlandaise, VBTI, maintient le projet en vie. Son [fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) republie toute la stack sous un préfixe `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` et les autres), recompilée pour les versions actuelles de PyTorch 2.x, CUDA et Python 3.10+, ce qui résout le blocage de version. Il est sous Apache-2.0 et activement maintenu, avec v3.5.1 en mai 2026. Pour profiter de la diversité de MMDetection sans le travail d'installation, utilisez ce fork. C'est une petite équipe, alors contribuez en retour si vous en dépendez.

Deux outils apparentés :

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) est en mode maintenance, sans version taguée depuis v0.6 en 2021, mais reste fiable et la source la plus propre de Mask R-CNN et de la famille R-CNN pour la segmentation d'instances et panoptique.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, activement maintenu) fournit Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN et Keypoint R-CNN. Pas de YOLO ou de DETR moderne, et vous écrivez votre propre boucle d'entraînement, mais c'est la base de comparaison sans dépendance supplémentaire.

Un point d'attention pour l'usage commercial : **MMYOLO**, le point de rassemblement des réimplémentations de YOLO d'OpenMMLab, est sous GPL-3.0 plutôt que sous Apache-2.0 et est figé depuis août 2023.

## 6. La famille RT-DETR

Licence : Apache-2.0 partout. Idéale pour la détection en temps réel à l'état de l'art avec du code de recherche.

La pointe actuelle de la détection en temps réel est un ensemble de détecteurs à base de transformers apparentés plutôt qu'un YOLO. La plupart descendent de RT-DETR, partagent une grande partie du code de backbone et d'encodeur, et sont presque tous sous Apache-2.0, ce qui facilite le passage de l'un à l'autre. La plupart se limitent à la détection, à une exception près (EdgeCrafter), qui étend les mêmes idées de distillation à la segmentation et à la pose. Leur coût commun tient à leur nature de dépôts de recherche : entraînement par fichiers de configuration, ergonomie différente d'Ultralytics et aucun canal d'assistance. Les modèles sont performants, et l'export ONNX et TensorRT est généralement très bien pris en charge.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Reformule la régression des boîtes en raffinement de distribution avec auto-distillation. Une forte exactitude par rapport à la latence (D-FINE-X atteint environ 55.8 AP en environ 13 ms sur un T4), des tailles de N à X et une intégration à Hugging Face Transformers en font le plus facile de cette liste à charger et à affiner en dehors de son propre dépôt.
- **[DEIM et DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) est une recette d'entraînement (appariement Dense O2O) qui se greffe sur D-FINE ou RT-DETRv2 et réduit à peu près de moitié le temps d'entraînement. DEIMv2 ajoute des caractéristiques DINOv3 et descend jusqu'à une gamme Atto de moins de 1M de paramètres pour le mobile ; DEIMv2-S est le premier modèle de moins de 10M à dépasser 50 AP. Il est activement maintenu, avec des mises à jour jusqu'en 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Le travail le plus récent du même laboratoire et le seul de cette liste qui dépasse la détection : il couvre la détection (ECDet), la segmentation d'instances (ECSeg) et la pose (ECPose), chacune en tailles S/M/L/X, toutes sous Apache-2.0. Il adapte un grand ViT pré-entraîné avec DINOv3 en un enseignant spécialisé par tâche, puis le distille en backbones élèves compacts conçus pour l'edge. Les résultats sont solides au regard de la taille : ECDet-S atteint 51.7 AP avec moins de 10M de paramètres sur COCO seul, ECPose-X atteint 74.8 AP (devant YOLO26Pose-X à 71.6), et sa segmentation d'instances rivalise avec celle de RF-DETR. Il est tout nouveau, donc considérez-le comme une version de recherche, mais c'est un des rares modèles compacts qui couvre les trois tâches.
- **[RT-DETR et RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Le travail original « DETRs beat YOLOs on real-time detection » (CVPR 2024) et la racine de cette famille : sans NMS, de bout en bout, facile à exporter. v2 est une mise à jour de 2024 de type bag-of-freebies, une amélioration directement substituable à latence égale. L'original Paddle et le portage PyTorch de référence partagent ce dépôt, et le modèle est intégré à HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) et **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Université de Pékin et Tsinghua, fin 2025). v3 ne prend en charge que PaddlePaddle. v4 est actuellement le meilleur de la famille (X proche de 57 AP), utilise PyTorch et distille un modèle de fondation visuel en un détecteur léger sans coût d'inférence supplémentaire. Son code reproduit aussi D-FINE, DEIM et RT-DETRv2 par des changements de configuration.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). Un DETR à base de ViT simple, présenté comme un remplacement de YOLO par un transformer, avec des tailles de tiny à xlarge et des exports ONNX et TensorRT. Il sert aussi de base à OVLW-DETR, à vocabulaire ouvert. L'activité récente est plus faible ; considérez-le comme une version de recherche stable.

LibreYOLO intègre déjà plusieurs d'entre eux (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) derrière une seule API, avec YOLO9 et RF-DETR comme modèles les plus testés. Pour les implémentations de référence et les derniers checkpoints de recherche, les dépôts sources ci-dessus font autorité.

## FAQ

**Ultralytics YOLO est-il gratuit pour un usage commercial ?**
Uniquement sous AGPL-3.0, qui exige de publier le code source de l'application que vous construisez avec, y compris les services accessibles en réseau. Un usage commercial à code fermé nécessite la licence commerciale payante d'Ultralytics. Les alternatives permissives (MIT, Apache-2.0) évitent cette contrainte.

**Quelle alternative à Ultralytics présente le moins de contraintes de licence ?**
Pour une stack permissive, déployable partout : LibreYOLO (code MIT), RF-DETR (Apache-2.0) et YOLOX (Apache-2.0). Tous évitent le copyleft de l'AGPL.

**Qu'est-ce qui remplace YOLO en 2026 ?**
De plus en plus, les détecteurs à base de transformers en temps réel : RT-DETR, RF-DETR et les familles D-FINE et DEIM. Ils égalent ou dépassent l'exactitude de YOLO à latence comparable sur du matériel suffisamment puissant. Les CNN de type YOLO gardent l'avantage sur les petits appareils edge, où ces transformers fonctionnent mal et manquent d'une solution NCNN ou CPU mature.

**Ces outils peuvent-ils fonctionner sur un Raspberry Pi ou un NPU ?**
Cela dépend de l'export. Les détecteurs CNN comme YOLOX et RTMDet s'exportent vers NCNN et fonctionnent sur un Pi, et certains (YOLOX) prennent en charge le NPU Hailo. Ce n'est pas le cas des détecteurs à base de transformers comme RF-DETR ; ils nécessitent un GPU ou un CPU puissant.

## Essayer LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO est sous licence MIT, fonctionne sur Linux, Mac et Windows et s'exécute sur GPU, Apple Silicon et simple CPU sans changer le code. Une seule API couvre RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter et d'autres, pour la détection, la segmentation, la pose, la classification, la profondeur, le regard et le suivi.

Ajoutez une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
