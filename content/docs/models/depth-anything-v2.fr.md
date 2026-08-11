---
title: Depth Anything V2
families:
  - depth_anything
seo_title: "Depth Anything V2\_: prédire et valider la profondeur monoculaire"
description: "Utilisez Depth Anything V2 dans LibreYOLO pour l'estimation de profondeur monoculaire. Installation, prédiction et validation\_; Small est publié sous Apache-2.0, Base et Large sous CC-BY-NC-4.0."
lead: "Depth Anything V2 est un encodeur DINOv2 associé à un décodeur DPT qui prédit une carte dense de profondeur inverse relative à partir d'une seule image. LibreYOLO le prend en charge pour la tâche de profondeur\_: prédiction et validation zero-shot, sans voie d'entraînement."
keywords:
  - Depth Anything V2
  - estimation de profondeur monoculaire
  - carte de profondeur python
  - DPT
  - DINOv2
  - profondeur relative
  - profondeur à partir d'une seule image
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Lire la carte de profondeur
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap : dense (H, W), valeur haute =
        plus proche

        raw = depth.data                # tenseur, sans unité métrique ni
        échelle commune

        normalized = depth.normalized() # ramené à [0, 1] pour la visualisation
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory route selon l'extension du fichier : un artefact exporté
        # se charge comme un checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Installation

Depth Anything V2 ne demande aucun extra optionnel. Tout ce qu'il importe fait
partie de l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

`result.depth_map` porte une carte dense de profondeur inverse relative : les
valeurs élevées correspondent aux points les plus proches de la caméra, et ces
valeurs n'ont ni unité métrique ni échelle commune d'une image à l'autre.
`save=True` écrit sur le disque une visualisation colorisée de cette carte ;
`Results.plot()` ne couvre pas cette famille, puisqu'il n'est défini que pour
les normales de surface et les contours. La résolution d'entrée doit être
divisible par 14, la grille de patchs DINOv2 sur laquelle s'appuie la tête
DPT ; LibreYOLO le vérifie avant l'exécution et lève une erreur si ce n'est pas
le cas. Voir [la prédiction](/docs/predict) pour les sources, le streaming et
le traitement des résultats.

## Variantes

Quatre tailles d'encodeur, s/b/l/g, correspondant à ViT-S/B/L/G. Le tableau des
checkpoints ci-dessous ne liste que s, b et l ; aucun checkpoint Giant n'est
publié. Les quatre partagent la même résolution d'entrée, si bien que choisir
une taille arbitre la capacité de l'encodeur, pas la taille d'image. La licence
compte aussi : le checkpoint Small est sous Apache-2.0, tandis que Base et
Large sont sous CC-BY-NC-4.0, voir Licence plus bas.

L'entraînement et le fine-tuning ne sont pas proposés pour cette famille.
`LibreDepthAnythingV2.train()` lève `NotImplementedError` sans condition ;
convertissez plutôt un checkpoint amont compatible, avec
`weights/convert_depth_anything_v2_weights.py`.

## Valider

`val()` exécute le validateur de profondeur commun : il aligne chaque prédiction
sur sa vérité terrain avec une échelle et un décalage estimés aux moindres
carrés, image par image, puis rapporte les métriques standard de profondeur
relative en zero-shot, AbsRel, RMSE et les trois seuils delta.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`, avec `depth_map` à la place des
boîtes. [L'export](/docs/export) liste les arguments que chaque format accepte.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
