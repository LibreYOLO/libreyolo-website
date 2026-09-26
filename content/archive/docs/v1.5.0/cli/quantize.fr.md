---
title: libreyolo quantize
seo_title: référence de la commande libreyolo quantize
description: "Quantifier un checkpoint en PyTorch depuis la ligne de commande\_: recettes, arguments de calibration, valeurs par défaut et familles acceptées par chaque recette."
lead: >-
  Remplace les modules float d'un modèle par des modules quantifiés, les calibre
  sur des images non étiquetées quand la recette a besoin de statistiques, et
  enregistre le résultat sous forme de checkpoint PyTorch.
keywords:
  - libreyolo quantize cli
  - quantification int8 ligne de commande
  - quantification fp8
  - quantification post entrainement yolo
  - arguments libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Commande
    value: libreyolo quantize
    mono: true
  - label: Requis
    value: model
    mono: true
  - label: Sortie
    value: >-
      Le chemin source avec -<recipe> avant le suffixe, p. ex.
      LibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: Basique
      language: bash
      code: |
        # Calibre sur coco128 et écrit LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Cast seul, sans calibration'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Calibration plus large, puis récupération'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # L'entraînement conscient de la quantification sur le checkpoint
        quantifié récupère l'exactitude.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Synopsis

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Les arguments sont des paires `key=value`, et la forme POSIX fonctionne aussi,
donc `recipe=int8` et `--recipe int8` sont le même argument.

## Arguments

| Argument | Par défaut | Signification |
|---|---|---|
| `model` | | Poids du modèle `.pt`. Requis |
| `recipe` | `int8` | Recette de quantification : `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Images de calibration : un YAML de données ou le nom d'un dataset intégré. Non étiquetées, forward uniquement. `none` saute la calibration |
| `samples` | `128` | Nombre maximal d'images de calibration |
| `batch` | `8` | Taille de batch de calibration |
| `algorithm` | `auto` | Estimation de la plage des activations : `auto`, qui sélectionne minmax, ou `minmax`, ou `percentile` |
| `out` | | Chemin du checkpoint de sortie. Par défaut, le chemin source avec `-<recipe>` avant le suffixe |
| `device` | `auto` | Périphérique |
| `allow_download_scripts` | `false` | Autorise le Python embarqué dans les blocs de téléchargement des YAML de dataset |
| `json` | `false` | Sortie JSON sur stdout |
| `quiet` | `false` | Supprime stderr |
| `help_json` | `false` | Affiche le schéma de la commande en JSON et quitte |

## Exemples

<code-tabs name="examples" />

## Notes

### Quelles familles l'acceptent

La quantification couvre quatre familles : `yolo9`, `rfdetr`, `birefnet` et
`feynobg`. Toute autre famille sort avec `quantize_failed`, qui porte la liste.

### Ce que touche chaque recette

`fp16` et `bf16` sont des casts. Ils changent seulement le dtype, n'ont besoin
d'aucune calibration, et `calib=none` est le réglage adapté pour eux.

`int8` et `fp8` quantifient les modules `Conv2d` et `Linear`, ce qui explique
qu'ils conviennent aux familles convolutives.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` et `int2` quantifient uniquement `nn.Linear`,
donc ils visent les familles transformer. Demander l'une d'elles sur `yolo9` est
refusé avec une explication plutôt que de produire en silence un modèle non
quantifié, car l'accélération en dessous de 8 bits n'y concerne que les GEMM et
les convolutions resteraient en précision supérieure.

`int8`, `fp8`, `w4a8` et `int2` ont besoin de statistiques de calibration pour
leurs activations. `int2` a aussi besoin d'un entraînement pour se rétablir
ensuite, donc il est refusé sur `birefnet` et `feynobg`, qui n'ont pas
d'entraîneur.

Chaque famille garde un ensemble de modules en float quelle que soit la recette :
les premières couches, les têtes de prédiction et, sur YOLOv9, la convolution
DFL, qui est un opérateur d'espérance intégrale fixe qui ne doit pas être
quantifié.

### Les données de calibration ne sont pas des données d'entraînement

`calib` pointe vers un petit ensemble d'images non étiquetées, utilisé uniquement
en forward, pour dériver les plages d'activation. Il ne sert pas à l'évaluation
et ses étiquettes ne sont jamais lues. Le `coco128.yaml` par défaut se télécharge
à la première utilisation depuis une URL, il ne demande donc aucune permission
supplémentaire ; un YAML avec un script de téléchargement Python embarqué a
besoin de `allow_download_scripts=true`.

`algorithm=percentile` est disponible et peut réduire l'exactitude sur les
familles transformer, ce qui explique que `auto` sélectionne minmax.

### Retrouver l'exactitude

La sortie est un checkpoint PyTorch normal, donc
[`libreyolo train`](/docs/cli/train) l'accepte directement. Entraîner un
checkpoint quantifié, c'est faire de l'entraînement conscient de la
quantification ; ajouter `distill_model=<teacher>` en fait de la distillation
consciente de la quantification.

### Sortie et codes de sortie

Le résultat affiche le chemin enregistré, la recette, le mode d'exécution, si la
calibration a été exécutée, et le nombre de modules remplacés par type. Le code
de sortie est `0` en cas de succès, `4` quand le modèle ne peut pas être chargé,
`5` quand la quantification ou l'enregistrement échoue, et `1` pour les autres
échecs à l'exécution.

Voir aussi : [`libreyolo export`](/docs/cli/export), qui quitte PyTorch et écrit
à la place un artefact de déploiement.
