---
title: Distillation des connaissances
seo_title: Distillation des connaissances dans LibreYOLO
description: >-
  Entraîner un petit détecteur avec un enseignant plus grand ou un backbone
  DINOv2 gelé : pertes MGD, CWD et MSE de caractéristiques, points de
  prélèvement et familles compatibles.
lead: >-
  La distillation ajoute un second terme de perte qui rapproche les cartes de
  caractéristiques intermédiaires de l'élève de celles d'un enseignant gelé.
  LibreYOLO prélève les caractéristiques avec des hooks de propagation. La tête
  et la perte propres à l'enseignant n'interviennent jamais.
keywords:
  - distillation connaissances
  - distillation générative masquée
  - distillation par canal
  - distillation caractéristiques
  - enseignant dinov2
  - entraînement enseignant élève
  - perte mgd
  - perte cwd
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

        # Un ViT auto-supervisé gelé supervise un étage du backbone.
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
    - label: Régler la perte
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
enseignant, chargé par la même fabrique que tout autre modèle.

<code-tabs name="detector" />

L'enseignant effectue sa propagation sous `no_grad` et sous autocast lorsque
l'AMP est activée. Le modèle gelé ne paie donc pas un calcul en pleine précision
à chaque étape. Des hooks de propagation capturent ses cartes de caractéristiques
à des points nommés, la perte les compare à celles de l'élève, puis le résultat
est ajouté à la perte d'entraînement et rapporté comme un composant nommé
`distill`.

## Distiller depuis un backbone fondamental gelé

Un ViT auto-supervisé peut à la place superviser un seul étage du backbone de
l'élève. Les caractéristiques de l'enseignant proviennent de son propre
extracteur plutôt que de hooks, et la perte gère la différence entre une grille
de patches et un pas convolutif.

<code-tabs name="foundation" />

`distill_model` reconnaît `dinov2`, qui correspond à DINOv2-base, ainsi que
`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`,
`dinov2-base`, `dinov2-large` et tout identifiant Hub brut commençant par
`facebook/dinov2`. Toute autre valeur est traitée comme le chemin d'un
checkpoint enseignant.

Ce parcours emploie `feat_mse` quelle que soit la valeur de
`distill_loss_type` et nécessite l'installation de `transformers`. Si
l'enseignant se charge avec des clés de poids manquantes, l'exécution s'arrête
au lieu de distiller depuis un backbone en partie aléatoire.

## Familles compatibles

La prise en charge de la distillation repose sur une méthode du modèle élève,
et deux méthodes existent.

`get_distill_config()` fournit les points de prélèvement multi-échelles
supervisés par un enseignant détecteur. YOLOv9, YOLOX et RF-DETR l'implémentent.

`get_backbone_distill_config()` fournit l'unique étage du backbone supervisé
par un enseignant fondamental. YOLOv9 l'implémente et constitue la seule famille
compatible.

Tout autre cas déclenche une erreur au lieu d'entraîner sans la perte :

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Points de prélèvement

Les points sont fixés par famille et par rôle. L'enseignant et l'élève n'ont
donc pas besoin de partager la même architecture, mais leurs pas de
caractéristiques doivent correspondre.

| Famille | Rôle | Points de prélèvement | Pas |
|---|---|---|---|
| YOLOv9 | enseignant ou élève | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | élève fondamental | `backbone.elan3` | 16 |
| YOLOX | enseignant ou élève | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | enseignant ou élève | `model.backbone.0.projector.stages.0` | sondé à la configuration |

Des pas incompatibles déclenchent une erreur avant le début de l'entraînement :

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Ce contrôle est ignoré pour les enseignants fondamentaux, dont l'objectif est
précisément de faire correspondre des grilles différentes.

## Trois pertes

`distill_loss_type` sélectionne la perte de caractéristiques pour un enseignant
détecteur. Un enseignant fondamental emploie toujours `feat_mse`.

`mgd`, la distillation générative masquée, masque une fraction des positions
spatiales de l'élève et entraîne un petit générateur à deux convolutions pour
reconstruire toute la carte de l'enseignant depuis les positions restantes.
`distill_mask_ratio` fixe la fraction masquée, 0,65 par défaut.

`cwd`, la distillation par canal, transforme les activations spatiales de chaque
canal en distribution de probabilités et minimise la divergence KL canal par
canal. `distill_tau` est la température du softmax, 1,0 par défaut.

`feat_mse` aligne les canaux de l'élève sur ceux de l'enseignant avec une
convolution 1 x 1, redimensionne bilinéairement la grille de l'enseignant vers
celle de l'élève et calcule l'erreur quadratique moyenne.
`distill_normalize=True` normalise d'abord les deux cartes en L2 sur la
dimension des canaux, ce qui limite la comparaison à l'angle et la rend
invariante à l'échelle. Sa valeur par défaut est `False`.

`dis` est le poids global appliqué par-dessus. S'il est omis, chaque perte
emploie sa valeur publiée : 2e-5 pour MGD, 1,0 pour CWD et 1,0 pour la MSE de
caractéristiques. Elles diffèrent de cinq ordres de grandeur. Un poids réglé
pour un type de perte n'a donc aucun sens pour un autre.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` et `distill_normalize` ne possèdent aucune
option CLI. Ce sont des arguments Python ou des clés YAML `cfg=`. Pour RF-DETR,
toute la distillation est également limitée à Python, car l'association
d'arguments de sa CLI ne transporte pas les clés de distillation.

## Adaptateurs, checkpoints et multi-GPU

Chaque perte construit de petits modules entraînables extérieurs à l'élève :
les adaptateurs de canaux 1 x 1 et le générateur de MGD. Ils reçoivent leur
propre groupe de paramètres dans l'optimiseur, au taux d'apprentissage effectif
de l'exécution.

Ces modules sont écrits dans le checkpoint sous une clé `distiller` et restaurés
à la reprise. Une exécution reprise ne redémarre donc pas avec des projecteurs
non entraînés.

Sous DDP, les adaptateurs restent en dehors de l'élève encapsulé. Le réducteur
DDP ne voit donc jamais leurs gradients. Le programme d'entraînement effectue
explicitement leur all-reduce à chaque étape afin que tous les rangs entraînent
les mêmes adaptateurs.

La capture de graphe CUDA n'est pas disponible pendant une distillation.
Transmettre `cuda_graph=True` journalise une ligne et poursuit en mode eager.
Consultez les [performances d'entraînement](/docs/train/performance).

## Voir aussi

- [Gel des couches](/docs/train/layer-freezing) et
  [fine-tuning LoRA](/docs/train/lora), qui peuvent tous deux être combinés à
  la distillation.
- [Hyperparamètres](/docs/train/hyperparameters) pour le reste de `train()`.
