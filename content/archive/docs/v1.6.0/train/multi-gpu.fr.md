---
title: Entraînement multi-GPU
seo_title: Entraînement multi-GPU dans LibreYOLO
description: >-
  Entraînez sur plusieurs GPU avec device="0,1". Le lancement des workers DDP
  par la bibliothèque, la raison pour laquelle batch est global, l'usage de
  sync_bn et le chemin torchrun.
lead: >-
  L'entraînement multi-GPU dans LibreYOLO utilise PyTorch
  DistributedDataParallel : un processus par GPU, chacun contenant une réplique
  complète du modèle et un fragment de chaque batch, avec les gradients moyennés
  entre les rangs à chaque étape.
keywords:
  - entraînement pytorch ddp
  - entraînement multi gpu
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - taille batch globale
  - backend nccl gloo
  - multi gpu windows
last_verified: 1.6.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Cette garde reste prise en charge ; les scripts ordinaires
        fonctionnent aussi sans elle.

        # Conservez-la pour les callbacks ou loggers qui nécessitent le repli
        pickle standard.

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch global : 16 images par GPU sur deux GPU
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
            # Sondé une fois sur le GPU 0, puis ajusté à un multiple du world size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: e339072d5d8e71ea
---

## Exécuter sur deux GPU

Passez une liste d'appareils. Rien d'autre ne change.

<code-tabs name="train" />

Lorsque plusieurs appareils sont fournis hors d'un environnement torchrun, la
méthode `train()` du modèle enregistre les poids dans un fichier temporaire,
résout l'autobatch s'il est demandé, puis démarre des processus workers gérés par un coordinateur, un par GPU. Chaque worker réimporte la classe du modèle, le
reconstruit depuis les poids enregistrés et exécute le chemin ordinaire à un
seul appareil, car les variables d'environnement torchrun sont définies depuis
un worker lancé. À la fin de l'exécution, le meilleur checkpoint du rang 0 est
rechargé dans l'instance du modèle de l'appelant.

`device` accepte `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` et
`"auto"`. Seule une liste de plusieurs indices CUDA déclenche le spawn.

## Lancement automatique et protection main

`model.train(device=[0, 1])` et `device="0,1"` utilisent des rangs gérés par un coordinateur sans réexécuter le code de niveau supérieur d'un script ordinaire sans protection. Les scripts protégés et `torchrun` explicite restent pris en charge. Les exécutions du coordinateur utilisent cloudpickle ; les objets callback ou logger qui exigent le repli vers pickle standard nécessitent toujours une protection `if __name__ == "__main__":`.

## batch est le batch global

`batch` est le nombre d'images par étape d'optimiseur sur tous les GPU. Le
dataloader de chaque rang est construit avec `batch // world_size` et un
`DistributedSampler`. `batch=32` sur deux GPU signifie donc 16 images par GPU,
pas 32.

Un batch qui n'est pas divisible par le world size provoque une erreur au lieu
d'entraîner silencieusement avec une autre taille :

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Les gradients sont moyennés par DDP lui-même, si bien que la loss est transmise
sans mise à l'échelle. La multiplier en plus par le world size augmenterait le
learning rate effectif d'environ un facteur égal au nombre de GPU.

## Autobatch sous DDP

`batch=-1` fonctionne et renvoie un batch global divisible par le world size.

<code-tabs name="autobatch" />

Sur le chemin spawn, la sonde s'exécute dans le processus parent sur le premier
appareil avant la création de tout worker. Chaque worker reçoit donc un entier
concret, sans nécessiter de coordination interprocessus. Sous torchrun, le rang
0 effectue la sonde et diffuse le résultat sous forme d'un unique tenseur long.

La sonde mesure la capacité d'un GPU et la multiplie par le world size. Lorsque
`nbs` est défini, le batch global est plafonné à `nbs` et arrondi vers le bas à
un multiple du world size. L'ajout de GPU réduit ainsi le nombre d'étapes
d'accumulation au lieu de réduire le batch par GPU. Le fonctionnement de la
sonde elle-même figure dans les
[hyperparamètres](/docs/train/hyperparameters).

## SyncBatchNorm

Sous DDP, les couches BatchNorm de chaque rang ne voient que leur propre
fragment. Avec `batch // world_size`, ce fragment peut devenir assez petit pour
que les statistiques courantes dégradent le modèle convergé par rapport à une
exécution sur un seul GPU.

`sync_bn=True` convertit chaque BatchNorm en SyncBatchNorm afin que les
statistiques soient calculées sur le batch global. La conversion ne se produit
que lorsque le mode distribué est actif. Une exécution sur un seul GPU n'est
donc jamais affectée par le flag.

Cette option est déjà activée par défaut pour les familles convolutionnelles
riches en BatchNorm : YOLOX, YOLOv7, YOLOv9 et ses variantes, YOLO-NAS, PicoDet,
RTMDet et FOMO. Toutes les autres familles la désactivent par défaut. Lorsqu'un
modèle contient une BatchNorm, que `sync_bn` est désactivé et que le batch par
rang est inférieur à 16, le trainer émet un avertissement.

<code-tabs name="syncbn" />

Il n'existe aucun flag CLI pour `sync_bn`. C'est un argument Python.

## Lancer avec torchrun

torchrun fonctionne aussi et constitue le bon choix lorsqu'un ordonnanceur de
cluster contrôle déjà le lancement des processus. Écrivez le script pour un
seul appareil et laissez torchrun définir l'environnement des rangs.

<code-tabs name="torchrun" />

Ne combinez pas les deux méthodes. En présence de l'environnement torchrun,
`device="0,1"` ne déclenche aucun spawn ; le trainer utilise `cuda:LOCAL_RANK`
et torchrun contrôle le nombre de processus.

## Comportement des rangs

Le rang 0 contrôle chaque effet de bord. Il résout le répertoire d'exécution et
diffuse le nom résolu afin que tous les rangs concordent, écrit les checkpoints
et les artefacts, puis déclenche les callbacks et les loggers de l'utilisateur.
Les autres rangs s'entraînent et contribuent aux gradients.

Chaque rang initialise son dataloader et le générateur aléatoire d'augmentation
différemment à partir de la valeur `seed` configurée, afin que les rangs ne
tirent pas des augmentations identiques.

## Plateforme et backend

Le backend est choisi automatiquement : NCCL lorsque CUDA et NCCL sont tous les
deux disponibles, Gloo sinon. NCCL n'est pas compilé sur Windows, les exécutions
Windows utilisent donc Gloo sans configuration. Le groupe de processus est
initialisé avec un délai maximal de trois heures.

## Éléments non exécutés sous DDP

- La capture de graphe CUDA. `cuda_graph=True` journalise une ligne et utilise
  le mode eager. Consultez les
  [performances d'entraînement](/docs/train/performance).
- Le profileur d'entraînement. `profile=True` est ignoré avec un avertissement.

Toutes les familles ne prennent pas le spawn automatique en charge. Vingt-quatre
le font, couvrant les familles de détection, de classification, de segmentation
sémantique et de restauration qui s'entraînent. Lorsqu'une famille ne le prend
pas en charge et reçoit un appareil multi-GPU, elle provoque une erreur qui
nomme l'API du modèle et la commande torchrun au lieu d'entraîner silencieusement
sur un seul GPU.

## Pages connexes

- [Hyperparamètres](/docs/train/hyperparameters) pour `batch`, `nbs` et la
  reprise.
- [Loggers d'expériences](/docs/train/loggers) pour la contrainte de
  sérialisabilité des callbacks.
- [GPU cloud](/docs/train/cloud-gpus) pour louer une machine multi-GPU.
