---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: "Depth Anything 3\_: prédire la profondeur monoculaire dans LibreYOLO"
description: >-
  Utilisez Depth Anything 3 dans LibreYOLO pour l'estimation de profondeur
  monoculaire. Installation, prédiction, validation et export du checkpoint
  DA3MONO-LARGE, sous Apache-2.0.
lead: "Depth Anything 3 est un simple transformer DINOv2 entraîné à prédire la profondeur et la géométrie de caméra à partir d'une ou plusieurs vues, sans aucune spécialisation architecturale. LibreYOLO porte son checkpoint DA3MONO-LARGE pour la tâche de profondeur\_: prédiction et validation zero-shot, sans chemin d'entraînement."
keywords:
  - Depth Anything 3
  - DA3
  - estimation de profondeur monoculaire
  - carte de profondeur python
  - profondeur relative depuis une image
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Lire la carte de profondeur
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap dense (H, W), plus grand = plus
        proche

        raw = depth.data                # tenseur, sans unité ni échelle commune

        normalized = depth.normalized() # ramené à [0, 1] pour la visualisation
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory route selon le suffixe du fichier, donc un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Installation

Depth Anything 3 ne demande aucun extra optionnel. Tout ce qu'il importe fait
partie de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

`result.depth_map` porte une carte dense de profondeur inverse relative : des
valeurs plus élevées signifient plus proche de la caméra, et ces valeurs n'ont
ni unité métrique ni échelle commune d'une image à l'autre. Le checkpoint
amont produit une profondeur relative positive ; le wrapper réseau de
LibreYOLO l'inverse et reproduit le traitement officiel du ciel, si bien que
la sortie respecte le contrat de profondeur partagé de LibreYOLO. `save=True`
écrit sur disque une visualisation colorisée de cette carte ; `Results.plot()`
ne couvre pas cette famille, puisqu'elle n'est définie que pour les normales
de surface et les contours. Voir [la prédiction](/docs/predict) pour les
sources, le streaming et le traitement des résultats.

## Variantes

Une seule taille, `l`, à une résolution d'entrée fixe. En amont, DA3 publie
aussi des checkpoints any-view Small et Base, un checkpoint de profondeur
métrique, ainsi que des checkpoints Nested et Giant ; LibreYOLO n'en expose
aucun. La profondeur métrique demande un contrat public différent de la tâche
de profondeur inverse relative de LibreYOLO, et les checkpoints any-view et
Nested demandent une API caméra multi-images que LibreYOLO ne propose pas. Les
checkpoints any-view Large et Giant sont par ailleurs sous CC-BY-NC-4.0 et ne
sont référencés par aucun chemin de téléchargement de LibreYOLO.

L'entraînement n'est pas proposé pour cette famille.
`LibreDepthAnything3.train()` lève `NotImplementedError` sans condition ;
entraînez en amont, puis convertissez un checkpoint DA3MONO-LARGE compatible
avec `weights/convert_depth_anything3_weights.py`.

## Valider

`val()` exécute le validateur de profondeur partagé : il aligne chaque
prédiction sur sa vérité terrain avec une échelle et un décalage estimés par
moindres carrés image par image, puis rapporte les métriques standard de
profondeur relative en zero-shot, AbsRel, RMSE et les trois seuils delta.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export est limité à cinq formats pour cette famille : ONNX, TorchScript,
ExecuTorch, TensorRT et OpenVINO. Demander tout autre format lève
`NotImplementedError` plutôt que de tenter une conversion non validée. Un
artefact exporté se recharge via `LibreYOLO()` d'après son suffixe de fichier,
si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un checkpoint et
renvoie le même objet `Results`, avec `depth_map` à la place des boîtes.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
