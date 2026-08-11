---
title: Hyperparamètres
seo_title: "Hyperparamètres d'entraînement dans LibreYOLO"
description: >-
  Les arguments importants de train() : epochs, batch, lr0, optimiseur, EMA,
  autobatch, accumulation de gradients et reprise, ainsi que la raison des
  valeurs par défaut propres à chaque famille.
lead: >-
  Chaque argument d'entraînement est un champ d'une dataclass TrainConfig. La
  classe de base définit le champ et sa valeur par défaut ; chaque famille de
  modèles en crée une sous-classe et remplace les valeurs par défaut modifiées
  par sa recette publiée.
keywords:
  - arguments entraînement
  - learning rate
  - taille de batch
  - autobatch
  - moyenne mobile exponentielle
  - accumulation gradients
  - reprendre entraînement
  - patience early stopping
  - amp bfloat16
  - configuration entraînement yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: "Lire les valeurs par défaut résolues d'une famille"
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Affiche les valeurs par défaut de train, val et predict, remplacements
        inclus.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sonde la mémoire GPU et choisit une puissance de deux
        concrète.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 4 micro-batchs de 16 par étape d'optimiseur, soit un batch effectif de
        64.

        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Chargez le checkpoint de l'exécution interrompue, puis demandez la
        reprise.

        model = LibreYOLO("runs/train/exp/weights/last.pt")

        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Les clés du yaml sont des champs TrainConfig. Les kwargs explicites
        gagnent.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Définir les arguments

`train()` accepte des arguments nommés, et la CLI accepte les mêmes noms sous
la forme `key=value`.

<code-tabs name="train" />

Les deux chemins aboutissent au même endroit. Les kwargs sont transmis à
`TrainConfig.from_kwargs()`, qui construit la dataclass de configuration de la
famille.

## Une faute de frappe ne provoque pas d'erreur

`from_kwargs()` élimine toute clé qui n'est pas un champ de la configuration et
émet un `UserWarning` qui la nomme. L'entraînement démarre alors avec la valeur
par défaut :

```python
# UserWarning : clés de configuration d'entraînement inconnues (ignorées) : ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Rien n'échoue, l'exécution se termine et le learning rate n'a jamais eu la
valeur demandée par l'appelant. Lisez les avertissements pendant la première
époque d'une nouvelle recette. La CLI est plus stricte, car elle valide les
noms des flags avant la construction de la configuration. Un flag CLI mal
orthographié est donc rejeté immédiatement.

## Valeurs par défaut propres à chaque famille

`TrainConfig` définit le champ et une valeur de base par défaut. Chaque famille
en crée une sous-classe et remplace ce que sa recette publiée modifie. Il
n'existe donc pas de réponse unique à la question « quel est le learning rate
par défaut ? ».

Les valeurs de base par défaut sont `optimizer="sgd"`, `lr0=0.01`,
`momentum=0.937`, `weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`,
`batch=16`, `imgsz=640` et `amp=True`. Voici trois exemples de l'écart d'une
famille par rapport à cette base :

| Champ | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE et DEIM sont fournis avec `amp=False`, car le décodeur D-FINE borne les
activations à 65504, la plus grande valeur float16 finie. YOLO-NAS et FOMO la
désactivent aussi par défaut. Le flag `--amp` de la CLI vaut `True` par défaut
pour chaque famille. Il est donc compté comme fourni par l'utilisateur et
remplace la valeur par défaut de la famille ; ne le modifiez que si c'est votre
intention.

Pour lire les véritables valeurs par défaut d'une famille au lieu de les
deviner :

<code-tabs name="defaults" />

## Taille de batch

`batch` est le batch global. Lors d'un entraînement multi-GPU, chaque rang
charge `batch // world_size`, si bien que le nombre transmis est le nombre
d'images par étape d'optimiseur quel que soit le nombre de GPU. Consultez
l'[entraînement multi-GPU](/docs/train/multi-gpu).

`batch=-1` active l'autobatch. Le trainer sonde le modèle en mode entraînement
avec une véritable passe backward sur des puissances de deux, ajuste une droite
à la courbe de mémoire et choisit la plus grande puissance de deux strictement
inférieure à la valeur extrapolée qui tient dans 60 % de la VRAM totale.

<code-tabs name="autobatch" />

Le point important est la sonde en mode entraînement avec une passe backward :
une sonde en mode inférence ne tient pas compte des activations conservées et
des tenseurs de gradients, qui représentent plusieurs fois l'empreinte de
l'inférence pour un CNN profond. RF-DETR abaisse la fraction cible à 45 %, car
la passe backward synthétique de la sonde sous-estime toujours le coût de son
critère et des couches auxiliaires du décodeur.

L'autobatch est une fonctionnalité CUDA. Sur CPU ou MPS, il journalise une ligne
et conserve le batch par défaut.

## Accumulation de gradients

`nbs` définit la taille de batch nominale, ou effective. Le trainer accumule
`round(nbs / batch)` micro-batchs par étape d'optimiseur.

<code-tabs name="accumulate" />

Avec la valeur par défaut `None`, l'accumulation est désactivée et
l'entraînement reste inchangé.

## Learning rate et schedule

`lr0` est le learning rate initial et `optimizer` accepte `sgd`, `adam` et
`adamw`. `momentum` correspond au momentum de SGD ou à beta1 d'Adam,
`weight_decay` est le terme L2 et `nesterov` s'applique à SGD.

Le schedule est déterminé par `scheduler`, `warmup_epochs`, `warmup_lr_start`
et `min_lr_ratio`. `no_aug_epochs` définit le nombre d'époques finales sans
augmentation forte, et plusieurs schedules l'utilisent aussi pour façonner
leur fin. Il ne s'agit donc pas uniquement d'un paramètre d'augmentation. Le
comportement de chaque famille pour sa partie augmentation figure dans la page
sur les [augmentations](/docs/train/augmentations).

Certaines familles ajoutent leurs propres paramètres de learning rate.
`backbone_lr_mult` met à l'échelle le groupe du backbone par rapport à la tête,
`clip_max_norm` définit l'écrêtage des gradients et SegFormer utilise
`head_lr_mult` pour exécuter sa tête de décodage à dix fois le learning rate du
backbone. Ces paramètres appartiennent à la sous-classe de configuration de la
famille, pas à la classe de base.

## EMA

`ema=True` conserve une moyenne mobile exponentielle des poids à côté des poids
entraînés. Elle est activée par défaut partout sauf pour FOMO.

`ema_decay` est le decay cible. Le decay augmente progressivement au lieu de
commencer à sa cible : sa valeur effective à la mise à jour `n` est
`ema_decay * (1 - exp(-n / tau))`, avec `tau` égal à 2000 par défaut. Les
premières mises à jour suivent donc le modèle de plus près, tandis que les
dernières le lissent. Les valeurs par défaut des familles vont de `0.997` pour
la pose YOLO-NAS à `0.9998` pour YOLOX et `0.9999` pour YOLOv9 et la lignée
DETR.

Les poids EMA sont ceux qui sont validés et stockés dans `best.pt` et `last.pt`.
Les poids entraînés bruts sont aussi stockés sous la clé `train_model`, afin
qu'une reprise continue depuis la trajectoire entraînée plutôt que depuis la
moyenne.

## Précision

`amp=True` exécute la passe forward sous autocast CUDA. `amp_dtype` sélectionne
`float16` (valeur par défaut) ou `bfloat16` ; `fp16` et `bf16` sont des formes
acceptées.

Float16 nécessite une mise à l'échelle dynamique de la loss et reçoit un
`GradScaler` actif. La plage d'exposants plus large de Bfloat16 n'en a pas
besoin, si bien que son scaler est construit mais désactivé, ce qui conserve un
chemin d'optimiseur identique. Demander bfloat16 sur un appareil CUDA qui ne le
prend pas en charge provoque une erreur pendant la configuration au lieu d'une
dégradation silencieuse.

## Sortie, checkpoints et arrêt

Les exécutions sont écrites dans `project/name`. `project` vaut `runs/train`
par défaut partout, mais `name` fait partie des remplacements propres aux
familles : la valeur de base par défaut est `exp`, tandis que YOLOv9 utilise
`yolo9_exp` et D-FINE `dfine_exp`. Avec `exist_ok=False`, la valeur par défaut,
un répertoire existant reçoit un suffixe incrémenté au lieu d'être écrasé.

`save_period` écrit un fichier `weights/epoch_<N>.pt` supplémentaire toutes les
N époques, en plus de `weights/last.pt` après chaque époque et de
`weights/best.pt` à chaque amélioration de la mesure suivie. `eval_interval`
définit la fréquence de la validation et `patience` arrête l'exécution après ce
nombre d'époques sans amélioration, `0` désactivant l'early stopping.

`cache` accélère les époques répétées en conservant les images décodées en RAM
(`True` ou `"ram"`) ou dans des fichiers `.npy` à côté des sources (`"disk"`).
Les lectures depuis le cache sont identiques octet par octet aux nouvelles
lectures. Avec des workers de dataloader, `"disk"` est le choix le plus sûr.

## Reprendre

`resume=True` poursuit une exécution interrompue. Le checkpoint doit d'abord
être chargé, car la reprise le lit depuis le modèle et non depuis un argument
séparé.

<code-tabs name="resume" />

La reprise restaure les poids entraînés, l'état de l'optimiseur, les poids EMA
et le nombre de mises à jour, le suivi de la meilleure mesure, l'échelle du
`GradScaler`, ainsi que les états aléatoires de PyTorch, CUDA et NumPy. Elle
commence à l'époque qui suit celle du checkpoint et avance le schedule jusqu'à
cette position.

Elle ne fait pas deux choses. `resume=True` ne peut pas être combiné à
`pretrained`, ce qui provoque une erreur. De plus, lorsque la clé de meilleure
mesure du checkpoint diffère de celle de l'exécution actuelle, son suivi est
remis à zéro avec un avertissement au lieu de comparer des valeurs qui n'ont
pas la même signification.

## Recettes dans un fichier

`cfg=` charge une correspondance YAML de noms de champs `TrainConfig` et la
fusionne sous les arguments nommés explicites, si bien qu'un kwarg l'emporte
toujours sur le fichier.

<code-tabs name="cfg" />

`size` et `num_classes` sont retirés du fichier, car l'instance du modèle les
possède déjà. Il n'existe aucun flag `--cfg` dans la CLI ; le chemin du fichier
est un argument Python.

## Pages connexes

- [Datasets](/docs/train/datasets) pour les valeurs acceptées par `data=`.
- [Augmentations](/docs/train/augmentations) pour les paramètres d'augmentation
  et les familles qui les respectent.
- [Gel des couches](/docs/train/layer-freezing) et [LoRA](/docs/train/lora) pour
  entraîner une partie des poids.
- [Validation et mesures](/docs/train/validation) pour les résultats rapportés
  par l'exécution.
