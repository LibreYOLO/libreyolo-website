---
title: Entraînement multi-GPU
seo_title: Entraînement multi-GPU dans LibreYOLO
description: >-
  Entraîner sur plusieurs GPU avec device="0,1". Comprendre la création des
  workers DDP, pourquoi batch désigne le lot global, quand définir sync_bn et
  comment utiliser torchrun.
lead: >-
  L'entraînement multi-GPU de LibreYOLO emploie DistributedDataParallel de
  PyTorch : un processus par GPU, chacun contenant une réplique complète du
  modèle et une partition de chaque lot, avec des gradients moyennés entre les
  rangs à chaque étape.
keywords:
  - entraînement pytorch ddp
  - entraînement multi gpu
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - taille lot globale
  - backend nccl gloo
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La garde __main__ est obligatoire : chaque worker créé réimporte ce
        # module. Sans la garde, il relancerait récursivement l'entraînement.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # lot global : 16 images par GPU avec deux GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Lancer
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Sondé une fois sur le GPU 0, puis ajusté à un multiple du nombre de processus.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Exécuter sur deux GPU

Transmettez une liste de périphériques. Rien d'autre ne change.

<code-tabs name="train" />

Lorsque plusieurs périphériques sont fournis sans environnement torchrun, la
méthode `train()` du modèle enregistre les poids dans un fichier temporaire,
résout le lot automatique s'il est demandé, puis crée un worker par GPU avec
`torch.multiprocessing.spawn`. Chaque worker réimporte la classe du modèle, le
reconstruit depuis les poids enregistrés et exécute le parcours ordinaire à un
périphérique, car les variables d'environnement torchrun sont définies depuis
un worker créé. Le meilleur checkpoint du rang 0 est rechargé dans l'instance
du modèle de l'appelant à la fin de l'exécution.

`device` accepte `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` et
`"auto"`. Seule une liste de plusieurs indices CUDA déclenche la création de
processus.

## Garde `__main__` obligatoire

Les workers créés réimportent le module dont ils proviennent. Sans garde
`if __name__ == "__main__":`, cette importation réexécute l'appel
d'entraînement et chaque worker crée ses propres workers. La bibliothèque
détecte ce cas et déclenche une erreur au lieu de laisser la récursion se
produire :

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Tout ce qui entre dans un worker est sérialisé par pickle. `callbacks=` doit
donc être sérialisable. Une classe au niveau du module fonctionne, contrairement
à une fermeture ou une lambda. Le message d'erreur le précise et renvoie vers
les systèmes de journalisation intégrés.

## `batch` désigne le lot global

`batch` est le nombre d'images par étape d'optimisation sur l'ensemble des GPU.
Le chargeur de données de chaque rang est construit avec
`batch // world_size` et un `DistributedSampler`. `batch=32` sur deux GPU
signifie donc 16 images par GPU, pas 32.

Un lot qui n'est pas divisible par le nombre de processus déclenche une erreur
au lieu d'entraîner silencieusement avec une autre taille :

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

DDP moyenne lui-même les gradients. La perte est donc transmise sans changement
d'échelle. La multiplier en plus par le nombre de processus augmenterait le
taux d'apprentissage effectif d'environ le nombre de GPU.

## Lot automatique sous DDP

`batch=-1` fonctionne et renvoie un lot global divisible par le nombre de
processus.

<code-tabs name="autobatch" />

Dans le parcours de création automatique, la sonde s'exécute dans le processus
parent sur le premier périphérique avant la création des workers. Chacun reçoit
donc un entier concret, sans coordination interprocessus. Sous torchrun, le
rang 0 effectue la sonde et diffuse le résultat comme un unique tenseur long.

La sonde mesure la capacité d'un GPU et la multiplie par le nombre de processus.
Lorsque `nbs` est défini, le lot global est limité à `nbs` et arrondi au
multiple inférieur du nombre de processus. Ajouter des GPU réduit ainsi le
nombre d'étapes d'accumulation au lieu de diminuer le lot par GPU. Le
fonctionnement de la sonde figure dans les
[hyperparamètres](/docs/train/hyperparameters).

## SyncBatchNorm

Sous DDP, les couches BatchNorm de chaque rang ne voient que leur propre
partition. Si `batch // world_size` est faible, les statistiques cumulées
peuvent dégrader le modèle convergé par rapport à une exécution sur un GPU.

`sync_bn=True` convertit chaque BatchNorm en SyncBatchNorm afin de calculer les
statistiques sur le lot global. La conversion n'a lieu que lorsque le mode
distribué est actif. L'option ne modifie donc pas une exécution à un GPU.

Elle est déjà activée par défaut pour les familles convolutives riches en
BatchNorm : YOLOX, YOLOv7, YOLOv9 et ses variantes, YOLO-NAS, PicoDet, RTMDet et
FOMO. Toutes les autres familles la désactivent par défaut. Lorsque le modèle
contient BatchNorm, que `sync_bn` est désactivé et que le lot par rang est
inférieur à 16, le programme d'entraînement affiche un avertissement.

<code-tabs name="syncbn" />

`sync_bn` ne possède aucune option CLI. Il s'agit d'un argument Python.

## Lancer avec torchrun

torchrun fonctionne également et convient lorsqu'un ordonnanceur de cluster
gère déjà le lancement des processus. Écrivez le script pour un seul
périphérique et laissez torchrun définir l'environnement des rangs.

<code-tabs name="torchrun" />

Ne combinez pas les deux parcours. En présence de l'environnement torchrun,
`device="0,1"` ne crée aucun processus. Le programme d'entraînement utilise
`cuda:LOCAL_RANK` et torchrun contrôle le nombre de processus.

## Comportement des rangs

Le rang 0 gère tous les effets de bord. Il résout le répertoire d'exécution et
diffuse son nom afin que tous les rangs soient d'accord, écrit les checkpoints
et les artefacts, puis déclenche les callbacks et les systèmes de
journalisation de l'utilisateur. Les autres rangs s'entraînent et contribuent
aux gradients.

Chaque rang initialise son chargeur de données et son générateur aléatoire
d'augmentations différemment à partir de la valeur `seed` configurée. Les rangs
ne tirent donc pas des augmentations identiques.

## Plateforme et backend

Le backend est choisi automatiquement : NCCL lorsque CUDA et NCCL sont tous
deux disponibles, Gloo dans les autres cas. NCCL n'est pas compilé sous
Windows. Les exécutions Windows emploient donc Gloo sans configuration. Le
groupe de processus est initialisé avec un délai d'expiration de trois heures.

## Fonctions indisponibles sous DDP

- Capture de graphe CUDA. `cuda_graph=True` journalise une ligne et poursuit en
  mode eager. Consultez les
  [performances d'entraînement](/docs/train/performance).
- Profileur d'entraînement. `profile=True` est ignoré avec un avertissement.

Toutes les familles ne prennent pas en charge la création automatique de
processus. Vingt-quatre la prennent en charge, couvrant les familles de
détection, de classification, de segmentation sémantique et de restauration
entraînables. Une famille incompatible à laquelle plusieurs GPU sont transmis
déclenche une erreur qui nomme l'API du modèle et la commande torchrun au lieu
de s'entraîner silencieusement sur un GPU.

## Voir aussi

- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `nbs` et la
  reprise.
- [Systèmes de journalisation des expériences](/docs/train/loggers) pour la
  contrainte de sérialisation des callbacks.
- [GPU dans le cloud](/docs/train/cloud-gpus) pour louer une machine multi-GPU.
