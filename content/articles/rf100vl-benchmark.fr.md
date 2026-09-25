---
title: "Benchmark RF100-VL : les modèles LibreYOLO généralisent-ils bien ?"
description: "Résultats RF100-VL mesurant la généralisation de YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter et RF-DETR sur 100 datasets réels après fine-tuning."
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL évalue la capacité d'un détecteur à s'adapter au-delà de COCO. Chaque modèle est affiné séparément sur 100 datasets très différents, notamment des images infrarouges, des radiographies, des images de microscopie, des scènes sportives, des images du spectre radio, des objets minuscules et des jeux vidéo. Nous avons évalué 17 configurations LibreYOLO : **1700 fine-tunings au total.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Résultats

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L arrive en tête de cette comparaison avec **61.76**. M suit avec 61.13, puis S avec 60.41. Les quatre résultats RF-DETR sont proches des chiffres publiés par [Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), ce qui valide utilement l'entraînement de RF-DETR dans LibreYOLO.

La taille ne prédit pas tous les résultats. YOLO-NAS-S et M sont pratiquement à égalité, avec 58.00 et 57.99. EdgeCrafter-M obtient 57.93, devant S à 55.99 et L à 56.11. Nous prévoyons d'étudier ces régressions. Cette campagne n'est pas une expérience contrôlée de mise à l'échelle : la résolution, les poids pré-entraînés, la précision et les recettes varient.

RF-DETR-S avec LoRA sur le backbone obtient 57.19, contre 60.41 pour le fine-tuning complet. Le fine-tuning complet l'emporte sur 95 datasets. Sa durée médiane enregistrée n'est que de sept minutes supérieure, tandis que la durée cumulée des tâches est presque identique.

Les lignes YOLOv9 datent d'avant d'importantes corrections de l'entraînement. Le résultat M anormal a contribué à révéler l'absence de PGI, une convention letterbox différente et une limite d'entraînement de 100 étiquettes. Les chiffres archivés décrivent le code qui a été exécuté. Ils ne constituent pas un verdict sur l'architecture YOLOv9.

Choisissez un modèle ci-dessous pour examiner ses scores sur les 100 datasets.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Méthodologie

Pour chaque dataset, nous avons entraîné le modèle sur `train`, sélectionné le meilleur checkpoint selon l'AP50:95 de validation, puis évalué ce checkpoint une seule fois sur `test`. Le score principal est la moyenne non pondérée de ces 100 résultats de test.

| Paramètre | Règle de la campagne |
|---|---|
| Entraînement | 100 époques ; sans early stopping |
| Batch effectif | 16 |
| Seed | 0 ; une exécution terminée par dataset |
| Checkpoint | Meilleur AP50:95 de validation avec les poids EMA |
| Score de test | pycocotools avec `maxDets=500` |
| Ajustement | Une recette fixe par famille ; aucun ajustement par dataset |

Chaque famille a conservé sa propre recette d'entraînement :

| Famille | Résolution | Précision | Grandes lignes de l'augmentation |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip ; augmentations fortes désactivées pendant les 15 dernières époques |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip ; les 15 dernières époques sans ces augmentations |
| YOLOX S/M | 640 | FP32 | Même recette pour la famille, avec un learning rate adapté à la taille |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV et flip ; mosaic désactivé |
| EdgeCrafter S/M/L | 640 | FP16 | Flip ; paramètres par défaut de la famille LibreYOLO |
| RF-DETR N/S/M/L et S LoRA | 384/512/576/704 ; LoRA à 512 | BF16 | Multi-échelle, recadrage/redimensionnement et flip |

Ces résultats reposent sur une seule seed. Les petites différences ne sont pas des améliorations établies. RF-DETR M et L ont utilisé l'accumulation de gradients pour tenir dans la mémoire disponible, et certains datasets denses ont nécessité des batchs physiques plus petits. Les recettes RF-DETR publiques partent également de checkpoints COCO, tandis que Roboflow a utilisé des checkpoints privés Objects365 pour son tableau historique.

Les durées d'entraînement sont des relevés de campagne, pas des benchmarks de vitesse contrôlés. Les tâches partageaient parfois des GPU, étaient relancées après un échec ou reprenaient après une interruption. Nous n'avons pas exécuté le protocole facultatif de latence sur un artefact unique T4. YOLO-NAS utilise les [poids pré-entraînés de Deci, sous licence distincte pour un usage non commercial](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).

## Améliorations apportées par la charge de travail

Les petits tests peuvent prouver que l'entraînement démarre. Ils ne reproduisent pas des semaines de fine-tuning continu sur de nombreuses architectures et de nombreux datasets. À cette échelle, les chemins lents, la pression mémoire et les problèmes de correction sont devenus difficiles à ignorer.

- **Chargement des images et validation.** Nous avons mis en cache le redimensionnement déterministe avant l'augmentation, réutilisé les workers et les buffers du validateur, et utilisé des CUDA graphs pour les passes forward de validation lorsque cela était possible. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graphs pendant l'entraînement.** La prise en charge de la capture est passée de YOLOv9 et RF-DETR à 24 familles. Nous avons aussi corrigé une condition de concurrence sur la mémoire épinglée du DataLoader et l'invalidation du graphe lors des transitions d'entraînement. Un test YOLOv9-T est passé de 428.4 à 367.7 secondes, avec le même AP rapporté. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Score COCO.** L'évaluation de datasets denses prenait parfois plus longtemps que l'entraînement. L'intégration de faster-coco-eval a évalué les prédictions enregistrées des 100 splits de test en 8.4 secondes au lieu de 131.4. Sur 1,400 valeurs de métriques, 1,381 étaient identiques bit à bit ; l'écart maximal était de 2.22e-16 et l'AP principal n'a pas changé. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR et chemins d'entraînement partagés.** Une mise en correspondance L1 plus rapide, moins de synchronisations avec le périphérique, AdamW fusionné et une mémoire du matcher plafonnée ont réduit un pas RF-DETR-S de 266 à 234 ms lors du test enregistré. La même investigation a amélioré cinq autres matchers, la journalisation YOLOv9 et la réutilisation des tenseurs d'EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Nous avons dirigé les calculs d'attention compatibles vers PyTorch SDPA, intégré une implémentation CUDA sous Apache-2.0 et corrigé l'exécution en précision mixte. Nous avons aussi écrit un kernel Triton d'attention déformable intégré au dépôt pour l'inférence. Dans les tests documentés, il était environ quatre fois plus rapide isolément et environ 7 % plus rapide de bout en bout pour RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Correction de l'entraînement.** Nous avons corrigé la reconstruction de BatchNorm de YOLOX et rétabli PGI, les métadonnées de letterboxing, la capacité des étiquettes et le warmup du momentum de YOLOv9. Le benchmark a fourni les checkpoints et les motifs d'échec qui ont révélé ces deux problèmes. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Ces chiffres proviennent de tests distincts sur des charges de travail différentes. Il ne faut pas les multiplier pour en tirer une seule accélération globale. LibreYOLO est désormais plus rapide et plus fiable sur des charges d'entraînement réelles qu'avant cette campagne.

## Harness et artefacts

Le [harness d'entraînement public](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) répartit les datasets entre les GPU, reprend les tâches interrompues, gère la récupération après les erreurs OOM et enregistre les recettes, les versions des datasets, les logs, les checkpoints et les prédictions.

Le [skill run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) fournit aux agents de programmation IA le protocole et les instructions d'utilisation de Vast.ai. L'[archive des résultats](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) contient chaque exécution publiée.

## Merci, Roboflow

Joseph Nelson a proposé une prise en charge GPU après que j'ai écrit que je souhaitais évaluer au-delà de COCO, mais que je ne pouvais pas financer les exécutions. Matvei Popov a partagé les configurations de référence, expliqué les paramètres d'évaluation et suivi l'arrivée des résultats.

Grâce à ce soutien, nous avons pu valider l'entraînement LibreYOLO sur 17 configurations, publier des éléments concrets pour aider au choix d'un modèle, publier le harness et mettre la bibliothèque à l'épreuve jusqu'à rendre ses points faibles évidents.

Merci à Joseph, Matvei et à l'équipe Roboflow pour les ressources de calcul, le temps et les conseils.
