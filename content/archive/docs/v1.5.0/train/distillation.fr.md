---
title: Distillation de connaissances
seo_title: Distillation de connaissances dans LibreYOLO
description: >-
  Entraînez un petit détecteur avec un teacher plus grand ou un backbone DINOv2
  gelé : les loss MGD, CWD et MSE de caractéristiques, les points d'extraction
  et la prise en charge par famille.
lead: >-
  La distillation ajoute un second terme de loss qui rapproche les cartes de
  caractéristiques intermédiaires du student de celles d'un teacher gelé.
  LibreYOLO extrait les caractéristiques avec des hooks forward, si bien que la
  tête et la loss propres au teacher n'interviennent jamais.
keywords:
  - distillation de connaissances
  - masked generative distillation
  - channel-wise distillation
  - distillation caractéristiques
  - teacher dinov2
  - entraînement teacher student
  - loss mgd
  - loss cwd
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un checkpoint plus grand de la même famille supervise le petit.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un ViT auto-supervisé gelé supervise une étape du backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Régler la loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # poids global de la distillation
            distill_tau=1.0,   # température du softmax CWD
        )
source_hash: 7210031328f6826f
---

## Distiller depuis un checkpoint plus grand

Définir `distill_model` active la distillation. Sa valeur est un checkpoint
teacher, chargé par la même factory que n'importe quel autre modèle.

<code-tabs name="detector" />

Le teacher exécute sa passe forward sous `no_grad`, et sous autocast lorsque
l'AMP est active, si bien que le modèle gelé ne paie pas le coût de calcul en
pleine précision à chaque étape. Des hooks forward capturent ses cartes de
caractéristiques aux points d'extraction nommés, la loss les compare à celles
du student, puis le résultat est ajouté à la loss d'entraînement et rapporté
comme un composant nommé `distill`.

## Distiller depuis un backbone fondamental gelé

Un ViT auto-supervisé peut à la place superviser une seule étape du backbone du
student. Les caractéristiques du teacher proviennent de son propre extracteur
de caractéristiques plutôt que de hooks, et la loss gère la différence entre
une grille de patches et un stride convolutionnel.

<code-tabs name="foundation" />

`distill_model` reconnaît `dinov2`, qui correspond à DINOv2-base, ainsi que
`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`,
`dinov2-base`, `dinov2-large` et tout identifiant de hub brut commençant par
`facebook/dinov2`. Toute autre valeur est traitée comme le chemin d'un
checkpoint teacher.

Ce chemin utilise `feat_mse` quelle que soit la valeur de `distill_loss_type`,
et exige l'installation de `transformers`. Un teacher qui se charge avec des
clés de poids manquantes interrompt l'exécution au lieu d'effectuer une
distillation avec un backbone partiellement aléatoire.

## Familles prises en charge

La prise en charge de la distillation est une méthode du modèle student, et il
en existe deux.

`get_distill_config()` fournit les points d'extraction multi-échelles supervisés
par un teacher détecteur. YOLOv9, YOLOX et RF-DETR l'implémentent.

`get_backbone_distill_config()` fournit l'unique étape du backbone supervisée
par un teacher fondamental. YOLOv9 l'implémente et constitue la seule famille à
le faire.

Tout autre cas lève une erreur au lieu d'entraîner sans la loss :

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Points d'extraction

Les points d'extraction sont fixes pour chaque famille et chaque rôle. Le
teacher et le student n'ont donc pas besoin d'utiliser la même architecture,
mais leurs strides de caractéristiques doivent correspondre.

| Famille | Rôle | Points d'extraction | Strides |
|---|---|---|---|
| YOLOv9 | teacher ou student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | student fondamental | `backbone.elan3` | 16 |
| YOLOX | teacher ou student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher ou student | `model.backbone.0.projector.stages.0` | sondé lors de la configuration |

Des strides différents provoquent une erreur avant le début de l'entraînement :

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Cette vérification est ignorée pour les teachers fondamentaux, dont le principe
même est que les grilles diffèrent.

## Les trois loss

`distill_loss_type` sélectionne la loss de caractéristiques pour un teacher
détecteur. Un teacher fondamental utilise toujours `feat_mse`.

`mgd`, la masked generative distillation, masque une fraction des positions
spatiales du student et entraîne un petit générateur à deux convolutions pour
reconstruire la carte de caractéristiques complète du teacher depuis ce qui
reste. `distill_mask_ratio` définit la fraction masquée, 0.65 par défaut.

`cwd`, la channel-wise distillation, transforme les activations spatiales de
chaque canal en une distribution de probabilité et minimise la divergence KL
canal par canal. `distill_tau` est la température du softmax, 1.0 par défaut.

`feat_mse` aligne les canaux du student sur ceux du teacher avec une convolution
1x1, redimensionne bilinéairement la grille du teacher à celle du student et
calcule l'erreur quadratique moyenne. `distill_normalize=True` normalise d'abord
les deux cartes de caractéristiques en L2 sur la dimension des canaux, ce qui
limite la comparaison à l'angle et la rend invariante à l'échelle. La valeur par
défaut est `False`.

`dis` est le poids global appliqué au-dessus. S'il n'est pas défini, chaque loss
utilise sa propre valeur publiée par défaut : 2e-5 pour MGD, 1.0 pour CWD et
1.0 pour la MSE de caractéristiques. Ces valeurs diffèrent de cinq ordres de
grandeur, si bien qu'un poids réglé pour un type de loss ne signifie rien pour
un autre.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` et `distill_normalize` n'ont aucun flag CLI.
Ce sont des arguments Python ou des clés YAML `cfg=`. La distillation RF-DETR
est elle aussi réservée à Python dans son ensemble, car sa correspondance
d'arguments CLI ne transmet pas les clés de distillation.

## Adaptateurs, checkpoints et multi-GPU

Chaque loss construit de petits modules entraînables extérieurs au student :
les adaptateurs de canaux 1x1 et le générateur de MGD. Ils reçoivent leur propre
groupe de paramètres dans l'optimiseur au learning rate effectif de l'exécution.

Ces modules sont écrits dans le checkpoint sous une clé `distiller` et restaurés
à la reprise, afin qu'une exécution reprise ne redémarre pas ses projecteurs à
froid.

Sous DDP, les adaptateurs se trouvent hors du student enveloppé, ce qui signifie
que le réducteur DDP ne voit jamais leurs gradients. Le trainer effectue
explicitement leur all-reduce à chaque étape, afin que chaque rang entraîne les
mêmes adaptateurs.

La capture de graphe CUDA n'est pas disponible pendant une exécution avec
distillation. Passer `cuda_graph=True` journalise une ligne et utilise le mode
eager. Consultez les
[performances d'entraînement](/docs/train/performance).

## Pages connexes

- [Gel des couches](/docs/train/layer-freezing) et
  [fine-tuning LoRA](/docs/train/lora), qui peuvent tous deux être combinés à
  la distillation.
- [Hyperparamètres](/docs/train/hyperparameters) pour le reste de `train()`.
