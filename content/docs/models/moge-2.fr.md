---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2 : prédire, valider et exporter des normales de surface'
description: >-
  Utilisez MoGe-2 dans LibreYOLO pour la prédiction dense de normales de
  surface. Installez, prédisez, validez et exportez les checkpoints officiels
  ViT-S, ViT-B et ViT-L.
lead: >-
  MoGe-2 est un modèle de géométrie monoculaire à une seule passe qui prédit un
  champ dense de normales de surface à partir d'une image RGB. LibreYOLO le
  prend en charge uniquement pour l'estimation des normales, au moyen des
  checkpoints officiels ViT-S, ViT-B et ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - estimation des normales de surface
  - géométrie monoculaire
  - carte de normales
  - prédiction dense
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3), vecteurs unitaires float32
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # degrés
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # pourcentage de pixels
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Installer

MoGe-2 ne nécessite aucun extra facultatif. Tous ses imports figurent dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés automatiquement à la première utilisation :
LibreYOLO récupère la taille correspondante directement depuis les checkpoints
officiels et la met en cache localement.

<code-tabs name="predict" />

MoGe-2 renvoie un champ dense plutôt qu'un ensemble de détections.
`result.boxes` est donc vide, et `conf`, `iou` ainsi que `max_det` n'ont aucun
effet. `result.normal_map` contient le résultat : un tableau `(H, W, 3)` de
vecteurs unitaires dans le repère de la caméra OpenCV, où `+x` pointe vers la
droite, `+y` vers le bas, `+z` vers l'intérieur de la scène, et une surface face
à la caméra vaut `(0, 0, -1)`. La prédiction d'une liste d'images exécute une
passe par image ; cette famille ne dispose d'aucun chemin rapide par batch
empilé. Consultez la [prédiction](/docs/predict) pour les sources, le streaming
et le traitement des résultats.

## Variantes

Trois tailles d'encodeur sont fournies sous forme de checkpoints distincts :
ViT-S, ViT-B et ViT-L, toutes à la même résolution d'entrée. Le banc d'essai de
LibreYOLO n'a pas mesuré cette famille, aucune valeur d'exactitude publiée ne
permet donc de les comparer ; choisissez la taille en fonction de votre propre
budget de calcul.

## Valider

`val()` mesure l'erreur angulaire par rapport à un dataset de cartes de normales
appariées : les images se trouvent à côté de fichiers PNG de normales 16 bits
ayant le même radical, avec un masque de validité facultatif afin que les pixels
de remplissage et non valides ne soient jamais comptés. La méthode renvoie
l'erreur angulaire moyenne et médiane en degrés, ainsi que le pourcentage de
pixels situés à moins de 11.25, 22.5 et 30 degrés.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export des normales utilise un contrat de runtime à résolution fixe et avec
un batch de 1 : `dynamic` et toute valeur de `batch` autre que 1 sont refusés,
et `imgsz` doit être divisible par la taille de patch de l'encodeur ViT, ce que
LibreYOLO vérifie avant le début de l'exécution. Un artefact exporté se recharge
par `LibreYOLO()` à partir de son suffixe de fichier. Un fichier `.onnx` se
comporte donc comme un checkpoint et renvoie le même objet `Results`.

<code-tabs name="export" />

## Licence

<provenance-box>

LibreYOLO ne copie pas ces checkpoints dans sa propre organisation.
`LibreYOLO("LibreMoGe2s-normal.pt")` télécharge la taille correspondante
directement depuis les dépôts Hugging Face officiels à une révision figée et
vérifie le fichier par rapport à une somme de contrôle SHA-256 enregistrée avant
de l'utiliser.

</provenance-box>

## Citation

<citation-block />
