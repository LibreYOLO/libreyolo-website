---
title: Hyperparamètres
seo_title: Hyperparamètres d'entraînement dans LibreYOLO
description: >-
  Les arguments importants de train() : epochs, batch, lr0, optimizer, EMA, lot
  automatique, accumulation des gradients et reprise, ainsi que les raisons des
  différences entre les valeurs par défaut des familles.
lead: >-
  Chaque argument d'entraînement est un champ d'une dataclass TrainConfig. La
  classe de base définit le champ et sa valeur par défaut. Chaque famille de
  modèles en hérite et remplace les valeurs modifiées par sa recette publiée.
keywords:
  - arguments train
  - taux apprentissage
  - taille lot
  - lot automatique
  - moyenne mobile exponentielle
  - accumulation gradients
  - reprendre entraînement
  - patience arrêt anticipé
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
    - label: Lire les valeurs par défaut résolues d'une famille
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
        # Affiche les valeurs par défaut de train, val et predict, y compris
        celles de la famille.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 sonde la mémoire GPU et se résout en une puissance de deux
        concrète.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-lots de 16 par étape d'optimisation, lot effectif de 64.
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


        # Les clés YAML sont des noms de champs TrainConfig. Les kwargs
        explicites l'emportent.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Définir les arguments

`train()` reçoit des arguments nommés et la CLI les mêmes noms sous la forme
`key=value`.

<code-tabs name="train" />

Les deux parcours aboutissent au même endroit. Les arguments sont transmis à
`TrainConfig.from_kwargs()`, qui construit la dataclass de configuration de la
famille.

## Une faute de frappe ne déclenche pas d'erreur

`from_kwargs()` supprime toute clé qui n'est pas un champ de la configuration
et émet un `UserWarning` qui la nomme. L'entraînement démarre ensuite avec la
valeur par défaut :

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Rien n'échoue, l'exécution se termine et le taux d'apprentissage n'a jamais pris
la valeur demandée. Lisez les avertissements de la première époque d'une
nouvelle recette. La CLI est plus stricte, car elle valide les noms des options
avant la construction de la configuration. Une option CLI mal orthographiée est
donc immédiatement refusée.

## Valeurs par défaut propres aux familles

`TrainConfig` définit le champ et une valeur de base. Chaque famille en hérite
et remplace ce que sa recette publiée modifie. Il n'existe donc aucune réponse
unique à la question du taux d'apprentissage par défaut.

Les valeurs de base sont `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` et `amp=True`. Trois exemples illustrent l'ampleur des écarts :

| Champ | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE et DEIM sont publiés avec `amp=False`, car le décodeur de D-FINE limite
les activations à 65 504, la plus grande valeur float16 finie. YOLO-NAS et FOMO
le désactivent également par défaut. L'option `--amp` de la CLI vaut `True` par
défaut pour chaque famille. Elle compte donc comme fournie par l'utilisateur et
remplace la valeur par défaut de la famille. Ne la modifiez que volontairement.

Pour lire les véritables valeurs d'une famille plutôt que de les deviner :

<code-tabs name="defaults" />

## Taille du lot

`batch` désigne le lot global. En entraînement multi-GPU, chaque rang charge
`batch // world_size`. La valeur transmise reste donc le nombre d'images par
étape d'optimisation quel que soit le nombre de GPU. Consultez
[l'entraînement multi-GPU](/docs/train/multi-gpu).

`batch=-1` active le lot automatique. Le programme d'entraînement sonde le
modèle en mode entraînement avec une véritable rétropropagation sur des
puissances de deux, ajuste une droite à la courbe de mémoire et sélectionne la
plus grande puissance de deux strictement inférieure à la valeur extrapolée qui
tient dans 60 % de la VRAM totale.

<code-tabs name="autobatch" />

L'intérêt réside précisément dans la sonde en mode entraînement avec
rétropropagation. Une sonde d'inférence omet les activations conservées et les
tenseurs de gradients, qui représentent plusieurs fois l'empreinte d'inférence
pour un CNN profond. RF-DETR réduit la fraction cible à 45 %, car la
rétropropagation synthétique de la sonde sous-estime encore le coût de son
critère et des couches auxiliaires du décodeur.

Le lot automatique est une fonctionnalité CUDA. Sur CPU ou MPS, il journalise
une ligne et conserve le lot par défaut.

## Accumulation des gradients

`nbs` définit la taille nominale, ou effective, du lot. Le programme
d'entraînement accumule `round(nbs / batch)` micro-lots par étape d'optimisation.

<code-tabs name="accumulate" />

Avec la valeur par défaut `None`, l'accumulation est désactivée et
l'entraînement reste inchangé.

## Taux d'apprentissage et planning

`lr0` est le taux d'apprentissage initial et `optimizer` accepte `sgd`, `adam`
et `adamw`. `momentum` désigne le momentum de SGD ou beta1 d'Adam,
`weight_decay` le terme L2, et `nesterov` s'applique à SGD.

Le planning est façonné par `scheduler`, `warmup_epochs`, `warmup_lr_start` et
`min_lr_ratio`. `no_aug_epochs` définit le nombre d'époques finales sans
augmentation forte. Plusieurs plannings l'emploient aussi pour façonner leur
fin. Il ne s'agit donc pas uniquement d'un réglage d'augmentation. Le
comportement de chaque famille pour l'autre moitié figure dans les
[augmentations](/docs/train/augmentations).

Certaines familles ajoutent leurs propres réglages de taux.
`backbone_lr_mult` met à l'échelle le groupe du backbone par rapport à la tête,
`clip_max_norm` fixe l'écrêtage des gradients et SegFormer emploie
`head_lr_mult` pour exécuter sa tête de décodage à dix fois le taux du backbone.
Ces champs appartiennent à la sous-classe de configuration de la famille, pas à
la classe de base.

## EMA

`ema=True` conserve une moyenne mobile exponentielle des poids à côté des poids
entraînés. Elle est activée par défaut partout sauf pour FOMO.

`ema_decay` est la décroissance cible. Elle augmente progressivement au lieu de
démarrer à sa cible. La valeur effective à la mise à jour `n` est
`ema_decay * (1 - exp(-n / tau))`, avec `tau` égal à 2 000 par défaut. Les
premières mises à jour suivent ainsi le modèle de plus près, tandis que les
dernières sont davantage lissées. Les valeurs par défaut des familles vont de
`0.997` pour la pose YOLO-NAS à `0.9998` pour YOLOX et `0.9999` pour YOLOv9
et la lignée DETR.

Les poids EMA sont ceux qui sont validés et stockés dans `best.pt` et `last.pt`.
Les poids entraînés bruts sont aussi stockés sous la clé `train_model`. Une
reprise continue ainsi depuis la trajectoire entraînée plutôt que depuis la
moyenne.

## Précision

`amp=True` exécute la propagation avant sous autocast CUDA. `amp_dtype` choisit
`float16`, valeur par défaut, ou `bfloat16`. Les formes `fp16` et `bf16` sont
acceptées.

Float16 exige une mise à l'échelle dynamique de la perte et reçoit un
`GradScaler` actif. La plage d'exposants plus large de bfloat16 n'en a pas
besoin. Son scaler est construit mais désactivé afin de conserver le même
parcours d'optimisation. Demander bfloat16 sur un périphérique CUDA incompatible
déclenche une erreur à la préparation au lieu d'une dégradation silencieuse.

## Sortie, checkpoints et arrêt

Les exécutions sont écrites sous `project/name`. `project` vaut
`runs/train` par défaut partout, mais `name` fait partie des valeurs propres aux
familles. La base emploie `exp`, YOLOv9 `yolo9_exp` et D-FINE `dfine_exp`.
Avec `exist_ok=False`, valeur par défaut, un répertoire existant reçoit un
suffixe incrémenté au lieu d'être écrasé.

`save_period` écrit un fichier `weights/epoch_<N>.pt` supplémentaire toutes les
N époques, en plus de `weights/last.pt` après chaque époque et de
`weights/best.pt` à chaque amélioration de la métrique suivie.
`eval_interval` définit la fréquence de validation. `patience` arrête
l'exécution après ce nombre d'époques sans amélioration, et `0` désactive
l'arrêt anticipé.

`cache` accélère les époques répétées en conservant les images décodées en RAM
(`True` ou `"ram"`) ou dans des fichiers `.npy` à côté des sources
(`"disk"`). Les lectures mises en cache sont identiques octet par octet aux
nouvelles lectures. Avec plusieurs workers du chargeur de données, `"disk"` est
le choix le plus sûr.

## Reprendre

`resume=True` continue une exécution interrompue. Le checkpoint doit d'abord
être chargé, car la reprise le lit depuis le modèle et non depuis un argument
séparé.

<code-tabs name="resume" />

La reprise restaure les poids entraînés, l'état de l'optimiseur, les poids et le
nombre de mises à jour EMA, le suivi de la meilleure métrique, l'échelle du
`GradScaler` et les états aléatoires PyTorch, CUDA et NumPy. Elle démarre à
l'époque suivant celle du checkpoint et avance le planning jusqu'à cette
position.

Deux opérations restent impossibles. `resume=True` ne peut pas être combiné à
`pretrained` et déclenche une erreur. Lorsque la clé de la meilleure métrique du
checkpoint diffère de celle de l'exécution actuelle, le suivi repart de zéro
avec un avertissement au lieu de comparer des valeurs qui n'ont pas le même
sens.

## Recettes dans un fichier

`cfg=` charge une association YAML de noms de champs `TrainConfig` et la
fusionne sous les arguments nommés explicites. Un argument nommé l'emporte donc
toujours sur le fichier.

<code-tabs name="cfg" />

`size` et `num_classes` sont retirés du fichier, car l'instance du modèle les
possède déjà. Il n'existe aucune option `--cfg` dans la CLI. Le chemin est un
argument Python.

## Voir aussi

- [Datasets](/docs/train/datasets) pour les formes acceptées par `data=`.
- [Augmentations](/docs/train/augmentations) pour les réglages d'augmentation et
  les familles qui les respectent.
- [Gel des couches](/docs/train/layer-freezing) et [LoRA](/docs/train/lora) pour
  entraîner un sous-ensemble des poids.
- [Validation et métriques](/docs/train/validation) pour les valeurs rapportées
  par l'exécution.
