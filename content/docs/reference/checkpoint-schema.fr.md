---
title: Schéma des checkpoints
seo_title: Schéma de métadonnées v1.0 des checkpoints LibreYOLO
description: "Métadonnées contenues dans chaque checkpoint .pt LibreYOLO\_: clés requises, ajouts par tâche, clés du runtime d'export, manifestes quantifiés et champs d'entraînement."
lead: "Un fichier .pt LibreYOLO est un dictionnaire plat enregistré avec torch.save. La clé model contient le state dict\_; les autres clés de premier niveau sont des métadonnées qui identifient le checkpoint sans analyser son nom de fichier ni inspecter son state dict."
keywords:
  - schéma checkpoint libreyolo
  - schema_version 1.0
  - model_family
  - métadonnées checkpoint libreyolo
  - manifeste quant
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Reproduit docs/checkpoint_schema.md du dépôt libreyolo en v1.5.0, recoupé avec
  libreyolo/utils/serialization.py et BaseModel.save.
snippets:
  usage:
    - label: Lire les métadonnées d'un checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Téléchargez un checkpoint, puis réenregistrez-le pour disposer d'un
        chemin local.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Schéma v1.0

Chaque checkpoint `.pt` officiel de LibreYOLO contient\u00a0:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Clé | Type | Signification |
|---|---|---|
| `model` | state dict | Poids du modèle |
| `schema_version` | str | Version du contrat de métadonnées\u00a0; la v1.0 utilise la chaîne `"1.0"` |
| `libreyolo_version` | str | Version ayant produit le checkpoint |
| `model_family` | str | Famille enregistrée, comme `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variante au sein de la famille, comme `t`, `s`, `r18`, `atto` |
| `task` | str | Nom canonique de la tâche |
| `nc` | int | Nombre de classes positif |
| `names` | dict | `dict[int, str]` dont les clés appartiennent à `0..nc-1` |
| `imgsz` | int | Résolution d'entrée carrée positive, ou ancien scalaire pour un contrat rectangulaire |

`task` est l'une des valeurs `detect`, `segment`, `semantic`, `panoptic`,
`pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`,
`restore`, `matte`, `ocr`, `embed` ou `mesh`.

Les checkpoints officiels écrivent chaque clé de `names`. Les lecteurs peuvent
compléter les clés manquantes avec des étiquettes `class_i` pour d'anciennes
tables clairsemées, mais les clés hors plage sont invalides.

Les checkpoints rectangulaires conservent un scalaire `imgsz` pour les anciens
lecteurs, défini sur `max(imgsz_h, imgsz_w)`, et écrivent aussi `imgsz_h` et
`imgsz_w` avec les dimensions réelles. Un lecteur compatible avec les champs
rectangulaires doit les préférer au scalaire. Les familles à contrat
rectangulaire fixe, comme HRNet pose, refusent les tailles incompatibles à
l'exécution.

Le schéma est délibérément plat et `model` est délibérément un state dict.

<code-tabs name="usage" />

## Ajouts pour la pose

La pose n'utilise généralement qu'une classe, `nc: 1` avec `person`, mais la
tête de pose YOLO-NAS prend aussi en charge la pose multi-classe avec un
squelette de points clés partagé. Dans ce cas, `nc` et `names` décrivent les
classes comme pour la détection. Les exports de pose dans le runtime émettent
des `scores` de forme `[batch, anchors, nc]`.

| Clé | Signification |
|---|---|
| `num_keypoints` | Nombre positif de points clés utilisé par la tête de pose |
| `keypoint_dim` | `2` pour les étiquettes `x,y` ou `3` pour `x,y,visibility`\u00a0; les sorties du modèle exposent toujours `x,y,visibility` |
| `oks_sigmas` | Sigmas OKS facultatifs par point clé\u00a0; la valeur par défaut de la tâche pour `num_keypoints` est utilisée si absente |
| `num_keypoints_per_class` | Nombre facultatif de points clés par classe pour les têtes de style GroupPose dont le tenseur de points clés est complété par classe\u00a0; `0` pour les classes sans point clé |

## Ajouts pour les maillages

Les checkpoints de maillage utilisent `task: "mesh"`, `nc: 1` et
`names: {0: "person"}`. La disposition des paramètres diffère entre les
modèles corporels, leurs dimensions sont donc consignées et jamais supposées.

| Clé | Signification |
|---|---|
| `body_model` | Paramétrage, comme `mhr`\u00a0; requis et utilisé pour interpréter tous les champs ci-dessous |
| `num_betas` | Nombre de coefficients d'identité et de forme\u00a0; 45 pour MHR |
| `num_body_pose` | Largeur du bloc des paramètres de pose corporelle\u00a0; 130 pour MHR. Vecteur plat et non un triplet par articulation, car les articulations du rig ont différents degrés de liberté |
| `num_vertices` | Nombre de sommets émis par le décodeur\u00a0; 18439 pour MHR |
| `num_joints` | Nombre d'articulations émises par le décodeur\u00a0; 127 pour MHR |
| `rotation_format` | Encodage des rotations, comme `euler_zyx` pour MHR ou `axis_angle`. Jamais déduit de la forme du tenseur, car un vecteur à 3 éléments est ambigu |

## Valeurs factices des tâches denses

Plusieurs tâches prédisent des cartes denses plutôt que des classes. Les
emplacements de type classe n'existent donc que pour la compatibilité du
schéma.

| Tâche | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Les prédictions de contours sont des cartes denses de probabilités float32
dans `[0, 1]`.

Les checkpoints de restauration peuvent ajouter `degradation`, une courte
étiquette de corruption comme `deblur`, `denoise` ou `super-resolution`\u00a0;
`dataset`, une étiquette de provenance comme `GoPro` ou `SIDD`\u00a0; et `scale`,
un facteur entier positif d'agrandissement entre sortie et entrée, par exemple
`4` pour un modèle de super-résolution x4. Une valeur absente ou égale à `1`
signifie que l'image restaurée conserve la résolution d'entrée. Le runtime
déduit également l'échelle de la famille et de la taille. `scale` est donc une
métadonnée de provenance et non une exigence de chargement.

## Ajouts pour l'OCR

La famille `ppocr` fournit un checkpoint composite par niveau dont le state
dict `model` contient deux sous-modèles dans les espaces de clés `det.*` et
`rec.*`.

| Clé | Signification |
|---|---|
| `charset` | Alphabet CTC complet dans l'ordre des indices de sortie\u00a0: l'indice 0 est le blanc CTC, suivi du dictionnaire de reconnaissance puis de l'espace. Les chargeurs doivent le lire dans le checkpoint, jamais dans un fichier annexe |
| `pipeline` | Valeurs par défaut du pipeline intégrées lors de la conversion\u00a0: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Les arguments du runtime peuvent les remplacer à chaque appel |
| `components` | Réservé aux étapes facultatives du pipeline comme l'orientation du document, la correction de perspective et la rotation des lignes de texte. Vide dans la v1 |

## Métadonnées du runtime d'export

Les artefacts exportés utilisent la même double écriture rectangulaire\u00a0:
`imgsz_h` et `imgsz_w` sont écrits à côté de l'ancien scalaire `imgsz`, et un
lecteur qui ne comprend pas les champs rectangulaires ne doit pas interpréter
silencieusement le scalaire comme un contrat carré.

La prise en charge rectangulaire dans le runtime dépend de la famille et du
format. Les exports des familles YOLO9, HRNet, NAFNet et Real-ESRGAN peuvent
utiliser des `imgsz_h` et `imgsz_w` non carrés dans les formats compatibles.
Les familles ou formats sans prise en charge rectangulaire explicite refusent
les métadonnées plutôt que de prétraiter ces artefacts comme s'ils étaient
carrés. Les exports HRNet sont des têtes de recadrage de personnes fixes,
batch 1 et FP32, où W32 accepte 256x192 et W48 384x288. Le détecteur de
personnes n'est pas intégré au graphe.

Les exports avec NMS intégrée peuvent ajouter les clés plates suivantes\u00a0:

| Clé | Signification |
|---|---|
| `nms` | Booléen sous forme de chaîne\u00a0; `"true"` signifie que le graphe contient une sortie avec post-traitement intégré |
| `nms_conf` | Seuil de confiance intégré à la sortie |
| `nms_iou` | Seuil IoU intégré à la sortie |
| `max_det` | Nombre maximal de lignes de détection post-NMS émises par la sortie intégrée |
| `nms_raw_output` | Booléen sous forme de chaîne\u00a0; `"true"` signifie que le graphe expose aussi une sortie auxiliaire brute du détecteur |

Pour les exports ONNX de détection YOLO9 avec `nms=true`, la sortie `0`
(nommée `output`) est le tenseur post-NMS autonome aux seuils d'export. Lorsque
`nms_raw_output=true`, la sortie `1` (nommée `raw`) est réservée aux backends
LibreYOLO afin qu'ils appliquent l'écrêtage natif sur le canevas d'origine et
la sémantique de `predict(conf=..., iou=..., max_det=...)` dans le runtime. Les
consommateurs tiers doivent utiliser la première sortie.

Les exports de pose peuvent ajouter `num_keypoints`\u00a0; `keypoint_dim`, dont les
exports bruts de style GroupPose peuvent utiliser des valeurs supérieures comme
`8` lorsque le tenseur inclut des champs de précision ou de logits de classes\u00a0;
`num_keypoints_per_class` sous forme de liste encodée en JSON, où les
emplacements de classes sans point clé doivent être conservés puisqu'ils
définissent le schéma\u00a0; et `pose_input`, où `"person_crop"` signifie que le
graphe accepte un recadrage déjà extrait et ne contient aucun détecteur. Les
exports HRNet exigent cette valeur.

Les exports de classification peuvent ajouter `crop_pct`, un ratio flottant
de recadrage central dont la cible de redimensionnement préalable est
`round(imgsz / crop_pct)` et qui vaut `0.875` par défaut, ainsi que
`interpolation`, `"bilinear"` ou `"bicubic"`, dont la valeur par défaut est
`"bilinear"`.

Les exports ExecuTorch écrivent les métadonnées plates dans un fichier annexe
`<program>.pte.json` requis. Le contrat v1 utilise CPU, FP32, un batch de 1 et
un canevas d'entrée fixe. Il exige en outre `executorch_version`,
`executorch_delegate` égal à `"xnnpack"` et un nombre positif
`executorch_delegate_partitions`. Le chargeur refuse un fichier annexe qui
déclare un autre delegate, des formes dynamiques ou une précision autre que
FP32.

Les exports MNN écrivent les métadonnées plates dans un fichier annexe
`<model>.mnn.json` requis. Le contrat v1 utilise CPU, FP32, la détection
uniquement et une forme d'entrée NCHW fixe. Il exige en outre `mnn_version`,
`mnn_backend` égal à `"cpu"`, des listes ordonnées non vides
`mnn_input_names` et `mnn_output_names`, `mnn_input_shape` comme quatre entiers
positifs dans l'ordre `[batch, channels, height, width]`, et `mnn_batch` égal
à `mnn_input_shape[0]`. Le chargeur refuse les métadonnées dynamiques, non
FP32, ne portant pas sur la détection, associées à une famille non compatible
ou dont les formes sont incohérentes.

Les fichiers `.pte` et `.mnn` sont des artefacts propres à des backends, et non
des checkpoints PyTorch.

## Checkpoints quantifiés

Un modèle quantifié ajoute une clé plate facultative, `quant`, contenant un
manifeste avec `schema`, `recipe`, `keep_high_precision`, `execution`, la
provenance de la calibration, `module_count` et `state`. Les manifestes FP8
peuvent aussi contenir `fp8_tensorwise_weights`, la liste exacte des noms de
modules `QuantLinear` dont l'échelle de poids porte sur le tenseur entier et
non sur chaque canal de sortie. Un chargeur qui rencontre `quant` reconstruit
la structure de modules quantifiés et la politique d'échelle avant
`load_state_dict`.

`state` distingue les deux formes d'artefacts.

`"prepared"`, valeur par défaut, contient des poids maîtres FP32 avec des
buffers d'échelle `_q_*` et peut être entraîné. Un lecteur sans prise en charge
de la quantification peut ignorer la clé `quant` et charger les poids maîtres
comme un modèle flottant.

`"finalized"` est la forme de déploiement écrite par `export(format="pt")`.
Les poids maîtres sont supprimés et chaque module quantifié contient des poids
compactés à la place\u00a0:

| Recette | Tenseurs compactés | Déquantification |
|---|---|---|
| int8 | `weight_packed` int8 à la forme d'origine des poids, `_q_w_scale` FP32 par canal | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn à la forme d'origine, `_q_w_scale` FP32 avec une entrée par canal de sortie | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, deux codes 4 bits par octet, nibble faible en premier, code `q + 8`\u00a0; `_q_w_gscale` FP32 `[out, ngroups]`, groupes de 128 le long de in_features | Échelle par groupe |
| int2 | Quatre codes 2 bits par octet, code `q + 2`, groupes de 64 | Échelle par groupe |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, code `sign<<3 \| E2M1 level`\u00a0; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`\u00a0; `_q_w_amax` FP32 par tenseur | `block_scale * amax / (448 * 6)` |
| mxfp4 | Comme nvfp4 mais avec des blocs de 32 éléments et `weight_block_exp` int8 `[out, ceil(in/32)]` en plus | `2 ** exponent` |

Les buffers de plage d'activations `_q_act_lo`, `_q_act_hi` et
`_q_calibrated` sont conservés pour int8. Le manifeste enregistre `remainder`,
`"fp16"` ou `"fp32"`, pour les tenseurs non quantifiés. Le décompactage
reproduit la simulation bit à bit. L'inférence finalisée correspond donc
exactement à l'inférence préparée sur l'appareil de finalisation. Cette
disposition est le contrat stable destiné aux exporteurs et runtimes externes.

## Checkpoints d'entraînement

Les checkpoints du trainer utilisent le même noyau de métadonnées requis et
peuvent ajouter des champs plats d'entraînement et de reprise\u00a0:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` indique si le `model` de premier niveau est lissé par EMA.
Lorsque l'EMA est activée, `train_model`, `ema` et `ema_updates` conservent
l'état de reprise. Les poids d'inférence publiés doivent rester légers et ne
doivent pas contenir l'optimiseur, l'époque, la configuration, la loss ou
l'état de reprise EMA, sauf s'ils sont délibérément distribués comme
checkpoints d'entraînement.

Pour assurer la compatibilité entre versions, les lecteurs acceptent les
anciens alias de meilleure métrique `best_mAP50_95`, `best_mAP50`,
`best_metric` et `best_metric_name`.

## Snapshots externes

Le schéma régit les fichiers `.pt` créés par LibreYOLO. Il ne renomme ni
n'encapsule les snapshots upstream multifichiers utilisés par les niveaux de
modèles distincts.

La taille `14b-a7b` de LibreMODUS est une exception explicite\u00a0: l'alias se
résout par `LibreVLM(...)` vers un répertoire de fichiers upstream épinglés.
LibreYOLO ne leur ajoute pas de métadonnées v1.0 et ne les republie pas sous
forme de fichier `.pt`.

## Poids anciens et étrangers

Les nouveaux writers appliquent une validation stricte et doivent produire des
métadonnées v1.0. Lorsque des métadonnées manquent ou sont incomplètes, les
anciens checkpoints ayant l'apparence de fichiers LibreYOLO sont chargés par
le chemin de compatibilité avec un avertissement et des instructions de
conversion, tandis que les checkpoints upstream étrangers sont dirigés vers
la conversion automatique. Consultez les
[checkpoints upstream](/docs/reference/upstream-checkpoints).

## Assistants

Les assistants du schéma se trouvent dans `libreyolo.utils.serialization`\u00a0:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` ne modifie rien et renvoie la liste des
erreurs. Avec `strict=True`, il lève plutôt `CheckpointMetadataError`.
`model.save(path)` est la méthode prise en charge pour écrire un checkpoint
conforme.
