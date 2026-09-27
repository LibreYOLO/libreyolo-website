---
title: libreyolo predict
seo_title: référence de la commande libreyolo predict
description: "Lancer l'inférence depuis la ligne de commande\_: chaque argument, sa valeur par défaut lue dans la définition de la CLI, et les options qui changent ce qui arrive sur stdout."
lead: "Exécute un modèle chargé sur une source et affiche les prédictions. La source peut être une image, un répertoire, une vidéo, une URL ou un flux en direct\_; le modèle peut être un checkpoint ou un artefact exporté."
keywords:
  - libreyolo predict cli
  - commande inférence libreyolo
  - prédiction yolo en ligne de commande
  - arguments libreyolo predict
  - sortie json libreyolo
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo predict
    mono: true
  - label: Requis
    value: source
    mono: true
  - label: Sortie
    value: >-
      Prédictions sur stdout. Avec save=true, fichiers annotés sous
      runs/detect/predict
snippets:
  examples:
    - label: Basique
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Enregistrer les images annotées
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Classes filtrées, JSON sur stdout'
      language: bash
      code: >
        # la classe 0 est person dans la liste COCO fournie avec le checkpoint.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Synopsis

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Les arguments sont des paires `key=value`. La même commande accepte aussi la
forme POSIX, donc `conf=0.4` et `--conf 0.4` sont interchangeables, et un
booléen écrit `save=true` devient `--save`. Les noms comportant un underscore
acceptent les deux orthographes : `max_det=50` et `--max-det 50` atteignent la
même option.

`libreyolo detect predict ...` est accepté et se comporte de façon identique ;
le mot de tâche est retiré avant l'analyse.

## Arguments

| Argument | Défaut | Signification |
|---|---|---|
| `source` | | Chemin d'image, répertoire ou URL. Requis |
| `model` | `yolox-s` | Nom ou chemin du modèle |
| `conf` | `0.25` | Seuil de confiance |
| `iou` | `0.45` | Seuil d'IoU du NMS |
| `imgsz` | | Taille de l'image en entrée : `640` (carrée) ou `480x640` (HxL). La taille d'entrée propre au modèle si non défini |
| `classes` | | Filtrer par identifiants de classe, par ex. `[0,2,5]`. Un entier seul est accepté |
| `max_det` | `300` | Nombre maximal de détections par image |
| `half` | `false` | Inférence FP16 (CUDA uniquement, nécessite la prise en charge du modèle) |
| `save` | `false` | Enregistrer les images annotées |
| `batch` | `1` | Images par passe avant pour les sources de type répertoire. Au-delà de 1, une vraie inférence par batch est utilisée sur les modèles qui la prennent en charge |
| `stream` | `false` | Produire les résultats au fil de l'eau. Activé automatiquement pour les webcams et les flux en direct |
| `stream_buffer` | `false` | Mettre en mémoire tampon chaque image du flux au lieu de ne garder que la plus récente |
| `vid_stride` | `1` | Traiter une image sur N dans une vidéo ou un flux en direct |
| `show` | `false` | Afficher les résultats vidéo et en direct ; `q` arrête |
| `tiling` | `false` | Inférence par tuiles pour les grandes images |
| `overlap_ratio` | `0.2` | Taux de recouvrement des tuiles |
| `output_path` | | Chemin de sortie explicite. Sinon `project/name` quand `save=true` |
| `color_format` | `auto` | Couleur en entrée : `auto`, `rgb`, `bgr` |
| `output_file_format` | | Format de sortie : `jpg`, `png`, `webp` |
| `device` | `auto` | Appareil : `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Modèle de détection de visages (chemin ou nom CLI). Requis pour les modèles d'estimation du regard |
| `gallery` | | Galerie de visages `.npz` produite par `libreyolo enroll`, pour identifier les visages. Modèles d'embedding de visages uniquement |
| `gallery_threshold` | `0.4` | Seuil cosinus pour une correspondance d'identité dans la galerie |
| `project` | `runs/detect` | Racine du répertoire de sortie |
| `name` | `predict` | Nom de l'expérience |
| `exist_ok` | `false` | Réutiliser un répertoire de sortie existant |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprimer stderr |
| `verbose` | `false` | Sortie stderr détaillée |
| `help_json` | `false` | Afficher le schéma de la commande en JSON et quitter |

## Exemples

<code-tabs name="examples" />

## Notes

Un artefact exporté se charge comme un checkpoint, donc
`model=weights/LibreYOLO9s.onnx` et `model=weights/LibreYOLO9s.engine` sont des
valeurs valides pour `model`. Trois options sont refusées sur ces runtimes
plutôt qu'ignorées : `tiling`, `overlap_ratio` et `output_file_format` quittent
avec `config_unsupported` quand un backend de runtime ne peut pas les honorer.

`half` fonctionne à l'inverse. Les runtimes exportés le reçoivent et s'exécutent
en FP16 ; l'inférence PyTorch native indique dans les logs qu'il a été ignoré et
continue en FP32.

Les modèles d'estimation du regard fonctionnent en deux étapes et n'ont pas de
détecteur propre, donc `face_detector` est requis pour eux. `gallery` ne
s'applique qu'aux modèles dont la tâche est `embed` ; le passer à autre chose
quitte avec `config_unsupported`.

stdout ne transporte que les résultats ; la progression, les avertissements et
les erreurs vont sur stderr. `json=true` affiche un objet JSON par invocation,
ou un par image en streaming, chacun portant `schema_version`. `quiet=true` fait
taire stderr. Les deux ensemble donnent à un lecteur machine un flux stdout
propre.

Le code de sortie est `0` en cas de succès, `2` pour une erreur d'usage ou de
configuration, `3` quand la source est introuvable, `4` quand le modèle ne peut
pas être chargé, et `1` pour les autres échecs d'exécution.

`help_json=true` affiche les paramètres, types, valeurs par défaut et options de
la commande en JSON sans rien exécuter, ce qui est le moyen fiable de relire ce
tableau depuis une version installée.

À voir aussi : [`libreyolo val`](/docs/cli/val) pour des métriques mesurées sur
un dataset, [`libreyolo export`](/docs/cli/export) pour produire les artefacts
de runtime cités plus haut.
