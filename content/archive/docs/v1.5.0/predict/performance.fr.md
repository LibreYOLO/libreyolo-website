---
title: Performances d'inférence
seo_title: Accélérer l'inférence dans LibreYOLO
description: >-
  Graphes CUDA, demi-précision, traitement par batch, inférence par tuiles et
  augmentation à l'inférence au moment de la prédiction, avec les véritables
  valeurs par défaut et les familles compatibles.
lead: "Cinq réglages de prédiction modifient le débit ou l'exactitude\_: rejeu d'un graphe CUDA, précision, traitement par batch, tuilage et augmentation à l'inférence. Chacun s'applique à un ensemble précis de familles, et deux d'entre eux coûtent en exactitude ou en latence au lieu d'en économiser."
keywords:
  - cuda graphs inférence pytorch
  - yolo batch inference python
  - inférence fp16
  - tiled inference petits objets
  - inférence par tuiles grandes images
  - test time augmentation détection
  - capture_graph
  - prédiction batch dossier
last_verified: 1.5.0
verification: "Valeurs par défaut des arguments lues dans InferenceRunner.__call__ dans libreyolo/models/base/inference.py. API des graphes CUDA lue dans BaseModel.capture_graph, graph_info, release_graphs et cuda_graph_scope dans libreyolo/models/base/model.py\_; activation par famille lue dans la variable de classe SUPPORTS_CUDA_GRAPH. Comportement en demi-précision lu dans NOOP_PREDICT_KWARGS dans libreyolo/utils/predict_args.py, l'avertissement du CLI dans libreyolo/cli/commands/predict.py, ainsi que CAST_RECIPES et SUPPORTED_FAMILIES dans libreyolo/quant/api.py. Conditions de traitement par batch lues dans InferenceRunner._process_in_batches et _predict_batch. Tuilage lu dans _predict_tiled et _merge_tile_detections. Augmentation à l'inférence lue dans BaseModel._predict_augment et _merge_tta, avec TTA_ENABLED, TTA_SCALES et TTA_FIXED_SIZE lus dans les modèles de libreyolo/models/."
snippets:
  batch:
    - label: Inférence par batch sur un dossier
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("batch_demo")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Une passe forward empilée par groupe de 4 sur les familles
        compatibles.

        results = model(str(folder), batch=4)

        print(len(results), "results")
    - label: 'Streaming, pour ne jamais matérialiser la liste'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: 'Capturer en amont, puis rejouer (CUDA requis)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Payez une fois le warmup et la capture, hors de la première requête.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Capturer uniquement lorsqu'une forme se répète (CUDA requis)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" attend qu'une forme apparaisse deux fois, une tâche ponctuelle
        # ne paie donc jamais le coût de la capture.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Installer l'extra d'export
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Exporter puis recharger avec la précision par défaut
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Export FP16 (construisez-le et exécutez-le sur une machine CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'FP16 dans PyTorch, avec une recette de conversion (CUDA requis)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Une recette de conversion ne lit aucune donnée de calibration.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Inférence par tuiles sur une grande image
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Le tuilage ne s'active que si l'image dépasse la taille d'entrée.
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Augmentation à l'inférence
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Réglages et valeurs par défaut

Chacun de ces réglages est un argument de `predict` et tous sont désactivés
par défaut.

| Argument | Valeur par défaut | Effet |
|---|---|---|
| `batch` | `1` | Nombre d'images par passe forward pour les sources dossier et liste |
| `cuda_graph` | `False` | Rejouer la passe forward depuis un graphe CUDA capturé |
| `tiling` | `False` | Diviser une grande image en tuiles qui se chevauchent |
| `overlap_ratio` | `0.2` | Chevauchement des tuiles lorsque `tiling` est activé |
| `augment` | `False` | Exécuter des vues retournées et les fusionner |
| `half` | | Accepté, accompagné d'un avertissement et ignoré |
| `device` | `None` | Déplacer le modèle avant la prédiction |

`imgsz` influe également sur le coût puisqu'il définit la résolution
d'exécution du modèle, mais il s'agit avant tout d'un argument d'exactitude et
il appartient au modèle plutôt qu'à cette page.

## Traitement par batch

<code-tabs name="batch" />

`batch` s'applique aux sources de type dossier et liste. Avec `batch=1`, chaque
image exécute sa propre passe forward. Au-delà de `1`, chaque groupe est
prétraité, empilé dans un tenseur unique, exécuté une fois, puis redécoupé afin
que le post-traitement mono-image existant de chaque famille reçoive ce qu'il
attend.

Le chemin empilé n'est emprunté que si toutes ces conditions sont remplies\u00a0:

- `batch` est supérieur à `1`
- `tiling` est désactivé
- l'augmentation à l'inférence n'est pas active
- la famille définit `SUPPORTS_BATCHED_PREDICT`
- le réseau sous-jacent n'est pas en mode entraînement

La dernière condition n'est pas un détail technique. Un réseau en mode
entraînement normaliserait le groupe empilé à l'aide de statistiques de batch
communes aux images. Les images d'un même groupe modifieraient alors
mutuellement leurs prédictions, ces exécutions restent donc séquentielles.

`SUPPORTS_BATCHED_PREDICT` vaut true par défaut. Les familles suivantes le
désactivent et exécutent une image par passe forward quelle que soit la valeur
de `batch`\u00a0: Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS,
HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet,
SAM 3D Body, SwinIR, YOLOv1, ZipDepth, tous les détecteurs à vocabulaire ouvert
et tous les modèles vision-langage.

Un autre mécanisme de repli existe. Si le prétraitement ne produit pas, pour
tout le groupe, des tenseurs `(1, C, H, W)` uniformes de même forme, dtype et
appareil, le groupe s'exécute séquentiellement au lieu d'être empilé. La
correction ne dépend ainsi jamais de dimensions d'images fortuitement égales.

Associez `batch` à `stream=True` sur un grand dossier pour bénéficier de passes
forward par batch sans conserver tous les résultats en mémoire.

## Graphes CUDA

<code-tabs name="graphs" />

Un graphe CUDA enregistre une fois une passe forward et la rejoue comme un
lancement unique. Les petits détecteurs consacrent une grande part de leur
temps de batch 1 au lancement des kernels. Regrouper ces lancements améliore
donc le débit et la sortie du rejeu est identique bit à bit à l'exécution eager.

`cuda_graph` accepte trois valeurs. `False` est la valeur par défaut et ne fait
rien. `True` effectue une capture lors de la première utilisation de chaque
forme d'entrée. `"auto"` attend qu'une forme se répète avant la capture, les
tâches ponctuelles et celles dont la forme varie ne paient donc jamais ce coût.

`capture_graph(imgsz=None, batch=1, dtype=None)` retire ce coût de la première
requête. Un graphe n'est valide que pour la forme exacte qu'il a capturée, la
valeur de `batch` doit donc correspondre à celle utilisée ensuite par
`predict`.

`graph_info()` rapporte les graphes capturés, les nombres de rejeux et toute
raison ayant provoqué un repli vers le mode eager. `release_graphs()` les
libère avec leurs buffers statiques.

La capture nécessite CUDA et une famille qui l'a activée avec
`SUPPORTS_CUDA_GRAPH`, car elle exige une passe forward sans opération visible
par l'hôte, vérifiée famille par famille. La demander sur une famille qui ne
l'a pas activée lève `NotImplementedError` au lieu d'exécuter silencieusement
le mode eager.

Un graphe enregistre des adresses mémoire et non des valeurs. Toute opération
qui déplace les paramètres le supprime donc. Le changement d'appareil au moyen
de `predict(device=...)`, la quantification et la déquantification invalident
tous les graphes capturés.

La matrice complète de prise en charge par famille, les séparations aux
jointures et le contrat numérique figurent dans la page sur les
[graphes CUDA](/docs/reference/cuda-graphs).

## Précision

<code-tabs name="precision" />

`half=True` au moment de la prédiction ne fait rien. Il est accepté pour la
compatibilité avec la ligne de commande, produit un avertissement indiquant
qu'il est sans effet, puis est retiré avant d'atteindre une famille. L'option
`--half` du CLI affiche le même avertissement pour un modèle `.pt`.

Deux véritables méthodes permettent de réduire la précision.

Pour un artefact exporté, la précision est choisie au moment de l'export avec
`export(format=..., half=True)`, et le fichier obtenu se recharge sans
modification par l'intermédiaire de `LibreYOLO()`.

Pour une exécution PyTorch, `model.quantize(recipe="fp16")` convertit le modèle
en float16 et installe des hooks qui conservent le float32 aux entrées et
sorties du modèle. `"bf16"` effectue la même opération en bfloat16. Aucune de
ces conversions ne lit de données de calibration, `calib` est donc ignoré.
La quantification couvre actuellement quatre familles\u00a0: YOLOv9, RF-DETR,
BiRefNet et FeyNobg. Une conversion sur un appareil CPU consigne un
avertissement de lenteur, ces recettes sont donc destinées à un GPU.

Les deux méthodes modifient les calculs numériques. Aucune ne garantit des
détections identiques sans autre intervention, effectuez donc une validation
avant le déploiement.

## Inférence par tuiles

<code-tabs name="tiling" />

Le tuilage découpe une grande image en tuiles carrées qui se chevauchent,
effectue une prédiction sur chacune, puis fusionne les résultats. Cette option
cible les petits objets dans les images haute résolution, lorsqu'un
redimensionnement de l'image entière réduit les cibles au-delà de la capacité
de résolution du modèle.

La taille des tuiles est la taille d'entrée du modèle, ou `imgsz` si vous la
fournissez, et elle doit être carrée. `overlap_ratio` vaut `0.2` par défaut.
Les tuiles qui se chevauchent sont réconciliées avec une suppression non
maximale par classe au seuil `iou`, puis la liste fusionnée est tronquée à
`max_det`. `iou` influe donc sur les prédictions par tuiles même pour les
familles qui n'exécutent aucune NMS elles-mêmes.

Le tuilage est omis, et pas simplement peu coûteux, si l'image tient déjà dans
la taille d'entrée\u00a0: si ses deux dimensions sont inférieures ou égales à cette
taille, une unique passe forward ordinaire s'exécute. Il est également omis
pour la classification, la segmentation sémantique et la tâche `embed`, qui se
rabattent sur une passe unique puisque le tuilage n'y a aucun sens.

Il lève une erreur pour les tâches dont la charge utile ne peut pas être
réassemblée\u00a0: masques de segmentation d'instances, bounding boxes orientées,
points, profondeur, contours et normales. Il ne peut pas être associé à
`augment`.

Le résultat contient `result.tiled` et `result.num_tiles`. Avec `save=True`,
les exécutions par tuiles écrivent un répertoire sous
`runs/tiled_detections` qui contient chaque tuile, l'image annotée, une
visualisation en grille et un fichier `metadata.json` enregistrant la taille
des tuiles, leur chevauchement et les seuils. `result.tiles_path` et
`result.grid_path` pointent vers ces éléments.

## Augmentation à l'inférence

<code-tabs name="tta" />

`augment=True` exécute l'image plusieurs fois et fusionne les détections avec
une suppression non maximale par classe au seuil `iou`. Comme le tuilage,
cette option rend `iou` déterminant pour les familles qui l'ignorent autrement.

En pratique, il s'agit d'un retournement horizontal. La liste d'échelles
`TTA_SCALES` contient par défaut une seule échelle de `1.0` et aucune famille
fournie ne la remplace. Chaque famille exécute donc deux passes\u00a0: l'image
d'origine et son reflet. Les familles marquées `TTA_FIXED_SIZE` redimensionnent
l'entrée vers un carré fixe, ce qui rend de toute façon le multi-échelle sans
effet pour elles.

La segmentation sémantique et la segmentation panoptique utilisent une fusion
différente. Leur vue retournée est remise dans le bon sens et les deux
distributions softmax sont moyennées avant l'argmax, au lieu d'être fusionnées
comme des bounding boxes.

L'augmentation à l'inférence n'est pas disponible pour toutes les tâches. Elle
lève une erreur pour les bounding boxes orientées, la pose, les points, la
profondeur, les normales, les contours, la restauration, l'OCR et les modèles
d'embeddings, et ne peut pas être associée au tuilage.

Les familles suivantes la désactivent entièrement, de sorte que `augment=True`
exécute une seule passe ordinaire\u00a0: BiRefNet, CenterNet, CLIP, DexiNed, FOMO,
HRNet, L2CS-Net, LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D
Body, SigLIP2, SwinIR, TEED, toutes les variantes de SAM, tous les détecteurs à
vocabulaire ouvert et tous les modèles vision-langage.

## Mesurer

Cette page ne fournit aucune mesure de latence, car un nombre de millisecondes
sans indication du matériel, du runtime, de la précision et de la taille de
batch n'est pas un fait. Des mesures sur différents matériels et runtimes sont
publiées sur [visionanalysis.org](https://www.visionanalysis.org), et
`libreyolo profile` mesure un modèle précis sur la machine devant vous.
