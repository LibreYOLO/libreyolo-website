---
title: libreyolo export
seo_title: référence de la commande libreyolo export
description: "Exporter un checkpoint vers un format de déploiement\_: chaque argument avec sa valeur par défaut, où atterrit l'artefact et les combinaisons que la commande refuse."
lead: >-
  Convertit un checkpoint en un format de déploiement et écrit l'artefact dans
  weights/. Le format décide lesquels des arguments ci-dessous s'appliquent.
keywords:
  - libreyolo export cli
  - exporter yolo en onnx
  - commande libreyolo export
  - export yolo tensorrt
  - arguments libreyolo export
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo export
    mono: true
  - label: Requis
    value: model
    mono: true
  - label: Sortie
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Basique
      language: bash
      code: |
        # Écrit weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS dans le graphe
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Exécuter l'artefact
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # La fabrique route sur le suffixe du fichier, donc l'export se charge
        comme un checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Synopsis

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi,
donc `format=onnx` et `--format onnx` sont le même argument.

## Arguments

| Argument | Par défaut | Signification |
|---|---|---|
| `model` | | Poids du modèle `.pt`. Requis |
| `format` | `onnx` | Format d'export : `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Plateforme cible RKNN, aujourd'hui `rk3588` uniquement. Rejeté avec tout autre format |
| `imgsz` | | Taille de l'image d'entrée : `640` ou `480x640` (HxW). `480,640` est aussi accepté. La taille propre au modèle si non défini |
| `batch` | `1` | Taille de batch de l'export |
| `half` | `false` | Précision FP16 |
| `int8` | `false` | Quantification INT8 |
| `dynamic` | `false` | Formes d'entrée dynamiques (ONNX) |
| `simplify` | `true` | Simplification du graphe ONNX |
| `nms` | `false` | Intégrer la NMS dans le modèle. ONNX et CoreML uniquement |
| `conf` | `0.25` | Seuil de confiance pour la NMS intégrée |
| `iou` | `0.45` | Seuil IoU pour la NMS intégrée |
| `max_det` | `300` | Nombre maximal de détections pour la NMS intégrée ONNX |
| `opset` | | Version de l'opset ONNX. Choisie automatiquement si non définie |
| `data` | | Données de calibration pour l'INT8 |
| `fraction` | `1.0` | Fraction des données de calibration à utiliser |
| `device` | `auto` | Appareil utilisé pour le tracing |
| `allow_download_scripts` | `false` | Autoriser le Python intégré dans les blocs de téléchargement des YAML de dataset |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprimer stderr |
| `verbose` | `false` | Journalisation détaillée de l'export |
| `verify` | `false` | Lancer le simulateur PC de RKNN Toolkit2 et comparer avec ONNX Runtime. RKNN uniquement |
| `help_json` | `false` | Afficher le schéma de la commande en JSON et quitter |

`engine` est un alias de `tensorrt` et `litert` un alias de `tflite`. Les deux
sont résolus vers le nom canonique avant que quoi que ce soit ne soit écrit, donc
la sortie JSON et la ligne de log indiquent toujours `tensorrt` ou `tflite`.

## Exemples

<code-tabs name="examples" />

## Notes

### Où atterrit le fichier

La commande ne prend pas de chemin de sortie. L'artefact est écrit dans
`weights/`, nommé d'après la racine du nom du checkpoint source plus le suffixe
du format, avec `_fp16` ou `_int8` inséré quand l'une de ces précisions a été
demandée. `LibreYOLO9s.pt` exporté en ONNX en FP16 devient
`weights/LibreYOLO9s_fp16.onnx`. Le résultat JSON porte le `output_path` résolu,
la taille du fichier en Mo, et la forme d'entrée sous la forme
`[batch, 3, height, width]`.

### Combinaisons refusées

`nms=true` est accepté pour ONNX et CoreML et refusé pour tous les autres formats
avec `nms_unsupported_format`. Sur ONNX, il force `dynamic` à false, puisque le
graphe intégré est figé à un batch de 1, et le signale sur stderr. Sur CoreML, il
prend `conf` et `iou` mais pas `max_det`, donc un `max_det` non par défaut à côté
de `format=coreml nms=true` se termine avec `config_unsupported`.

`half=true` en même temps que `int8=true` n'est pas une erreur. L'INT8 l'emporte,
`half` est abandonné, et un avertissement part sur stderr.

`name` et `verify` sont aujourd'hui des options RKNN. Passer l'un ou l'autre avec
un autre format se termine avec `config_unsupported` plutôt que d'être ignoré.

### Quels formats une famille prend en charge

La prise en charge est par famille et par tâche, pas globale. `libreyolo formats
family=<family> task=<task>` affiche le niveau de chaque format pour cette
combinaison, avec la raison et la contrainte éventuelle qui s'y rattache. Voir
[`libreyolo formats`](/docs/cli/utilities) pour les arguments.

Certains formats demandent une installation optionnelle et d'autres une
toolchain. Une dépendance Python manquante se termine avec `export_dep_missing` ;
une précision que le format ne peut pas produire se termine avec
`format_precision_unsupported`.

### Exécuter ce que vous avez exporté

Les artefacts exportés se chargent via la même fabrique de modèles que les
checkpoints, indexée sur le suffixe du fichier, donc `libreyolo predict
model=weights/LibreYOLO9s.onnx` fonctionne sans conversion supplémentaire. Trois
options de prédiction font exception et sont refusées sur les backends runtime :
`tiling`, `overlap_ratio` et `output_file_format`.

Deux cibles de déploiement ont leur propre page :
[NVIDIA DeepStream](/docs/export/deepstream) et
[NVIDIA Jetson](/docs/export/jetson).

### Sortie et codes de retour

stdout porte le résultat ; la progression part sur stderr. Le code de retour est
`0` en cas de succès, `2` pour une erreur d'usage ou de configuration, `4` quand
le modèle ne peut pas être chargé, `5` pour un format inconnu, une dépendance
d'export manquante, une précision non prise en charge ou une demande de NMS
intégrée refusée, et `1` pour les autres échecs à l'exécution.

À voir aussi : [`libreyolo quantize`](/docs/cli/quantize), qui reste dans PyTorch
et écrit un checkpoint plutôt qu'un artefact de déploiement.
