---
title: Restauration d'images
seo_title: Restauration et agrandissement d'images dans LibreYOLO
description: >-
  Débruiter, déflouter et agrandir des images dans LibreYOLO. Prédire une image
  RVB restaurée, entraîner NAFNet sur des données appariées et lire les clés
  PSNR et SSIM.
lead: >-
  La restauration d'images reçoit une image dégradée et en renvoie une version
  propre. LibreYOLO l'expose comme la tâche restore, qui couvre le débruitage,
  le défloutage et la super-résolution derrière un même contrat de sortie : une
  image RVB en entrée, une image RVB en sortie.
keywords:
  - restauration image python
  - modèle débruitage image
  - super résolution image python
  - modèle défloutage
  - validation PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Agrandir une image
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Le générateur 4x compact ; tile limite le pic de mémoire sur une
        grande source.

        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")

        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)


        result.restored.save("upscaled.png")

        print(result.restored.array.shape)   # 4x l'entrée sur chaque axe
    - label: Débruiter une image
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Entraîné sur le bruit d'images réelles SIDD ; la sortie conserve la
        taille d'entrée.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        result = model(SAMPLE_IMAGE)


        result.restored.save("denoised.png")

        print(result.restore_scale)   # 1 : aucun agrandissement pour ce
        checkpoint
  train:
    - label: Affiner NAFNet sur des images appariées
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Enregistrer la provenance dans le checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation et dataset sont inscrits dans le checkpoint enregistré
        # à des fins de provenance ; ils ne participent pas à l'entraînement.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Valider et lire les clés des métriques
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() renvoie un dictionnaire ordinaire et non un objet.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Exporter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz est fixé dans le graphe. Transmettez donc la taille que votre
        # déploiement fournit réellement au modèle.
        model.export(format="onnx", imgsz=256)
    - label: Exécuter le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La fabrique s'oriente grâce au suffixe du fichier. Un artefact exporté
        # se charge donc comme tout checkpoint et renvoie le même objet Results.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Définition

La tâche `restore` transforme une image en une autre. Le débruitage, le
défloutage et la super-résolution constituent ici la même tâche, car ils
partagent un contrat : le modèle reçoit une image RVB et renvoie une image RVB.
La dégradation qu'il a appris à corriger est une propriété du checkpoint plutôt
que de l'API.

Une prédiction remplit `result.restored`, une charge utile `RestoredImage` qui
contient un tableau RVB uint8 `(H, W, 3)`. `.array` le renvoie sous forme NumPy
et `.save(path)` l'écrit sur le disque. `result.restore_scale` enregistre le
facteur d'agrandissement du canevas de sortie. Il vaut `1` pour un checkpoint
qui préserve la résolution. `result.boxes` reste vide. Les arguments `conf`,
`iou` et `max_det` sont acceptés pour conserver la parité de la signature, mais
n'ont aucun effet. `save=True` écrit directement l'image restaurée plutôt
qu'une photo annotée.

## Modèles

Trois familles couvrent `restore`, réparties selon la dégradation qu'elles
annulent.

[NAFNet](/docs/models/nafnet) est le modèle de débruitage et la seule famille de
restauration que LibreYOLO peut entraîner. Son architecture remplace les
activations non linéaires d'un bloc UNet par une multiplication terme à terme.
Le checkpoint publié est entraîné sur le bruit d'images réelles SIDD. La sortie
conserve la résolution d'entrée.

[Real-ESRGAN](/docs/models/real-esrgan) est le modèle d'agrandissement pratique.
Ses trois checkpoints sont entraînés sur des dégradations synthétiques plutôt
que sur le seul sous-échantillonnage bicubique. Ils couvrent les facteurs 4x,
2x et un générateur 4x plus petit et plus rapide, conçu pour réduire la latence.

[SwinIR](/docs/models/swinir) agrandit les images 4x avec un backbone Swin
Transformer. Ses trois tailles couvrent le générateur léger officiel et deux
générateurs pour les images réelles.

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et mis
en cache localement.

<code-tabs name="predict" />

La restauration s'exécute à la propre résolution de l'image source plutôt que
sur un canevas réseau fixe. Elle ajoute seulement le remplissage nécessaire au
facteur de sous-échantillonnage du réseau. Le temps et la mémoire évoluent donc
avec le nombre de pixels de l'entrée. `tile` divise la propagation en tuiles qui
se chevauchent et fusionne les raccords. `tile_pad` est la marge ajoutée autour
de chaque tuile avant de la recadrer. Tous deux sont des arguments nommés Python.
Consultez la page [prédiction](/docs/predict) pour les sources, le streaming et
la gestion des résultats.

## Format du dataset

La restauration associe chaque image d'entrée dégradée à une image cible propre
de résolution exactement identique, mise en correspondance par le nom de base
du fichier.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` et `names` sont des valeurs réservées du schéma. Un modèle de restauration
renvoie `Results.restored`, pas des détections. `degradation` et `dataset` sont
des étiquettes facultatives de provenance. `target_stem_suffix` couvre les
datasets qui nomment l'image propre différemment de l'image dégradée qui lui est
associée. La validation conserve la résolution native et ajoute seulement le
remplissage nécessaire à l'empilement d'un lot. Les métriques sont donc
calculées sur le canevas d'origine. Consultez les
[formats de datasets](/docs/reference/dataset-formats) pour le contrat complet.

## Entraîner

NAFNet est la seule famille de restauration dotée d'une implémentation
d'entraînement. `Real-ESRGAN.train()` et `SwinIR.train()` déclenchent tous deux
une `NotImplementedError`. Ces checkpoints proviennent d'un entraînement GAN
sur des pipelines de dégradations synthétiques, et le programme d'entraînement
apparié de restauration s'exécuterait sans reproduire cette recette.

<code-tabs name="train" />

Le programme d'entraînement prélève des recadrages couplés dans les paires
d'entrée et de cible afin que les deux côtés restent alignés. Consultez la page
[entraînement](/docs/train) pour les datasets, le multi-GPU et les systèmes de
journalisation, ainsi que la [page NAFNet](/docs/models/nafnet) pour les valeurs
par défaut de cette famille et le pooling d'inférence qu'elle détache pendant
l'entraînement.

## Valider

`val()` compare en RVB la sortie restaurée à la cible propre, sur le canevas
d'origine, sans recadrage des bords ni redimensionnement.

<code-tabs name="val" />

`metrics/PSNR` est le rapport signal sur bruit de crête en décibels. Il sert
également de `fitness`, la valeur lue pour sélectionner le meilleur checkpoint.
`metrics/SSIM` est la similarité structurelle dans `[0, 1]`, calculée avec une
fenêtre gaussienne de 11 x 11 à sigma 1,5 et moyennée sur les trois canaux de
couleur. Pour les deux métriques, une valeur élevée est préférable.

## Exporter

Un modèle de restauration exporté se recharge par `LibreYOLO()` grâce au
suffixe de son fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme
un checkpoint et renvoie le même `Results`, où `restored` contient l'image de
sortie.

<code-tabs name="export" />

L'exportation de restauration fixe la résolution spatiale dans le graphe.
Transmettez donc la valeur `imgsz` que votre déploiement fournira réellement au
modèle. Pour NAFNet, cette taille doit être divisible par le facteur de
sous-échantillonnage du réseau, et seule la dimension du lot reste dynamique
avec `dynamic=True`. Pour Real-ESRGAN et SwinIR, omettre `imgsz` revient à une
petite taille de patch interne plutôt qu'à votre résolution de travail. La
couverture de chaque format figure sur les pages des modèles et dans la
[matrice d'exportation complète](/docs/reference/export-matrix).
La page [Exporter](/docs/export) présente les arguments acceptés par chaque
format.
