---
title: Gel des couches
seo_title: "Geler des couches pendant l'entraînement dans LibreYOLO"
description: >-
  Gelez une partie d'un modèle pour l'apprentissage par transfert : un nombre
  entier de groupes de gel propres à la famille, une liste explicite d'indices
  ou des sélecteurs de noms de modules et de paramètres.
lead: >-
  Le gel maintient les poids sélectionnés fixes tandis que le reste du modèle
  s'entraîne. Les sélecteurs ciblent les groupes de gel ordonnés ou les noms de
  modules propres à une famille, pas les numéros de couches bruts d'un graphe
  YAML.
keywords:
  - geler couches modèle
  - apprentissage par transfert
  - geler backbone
  - batchnorm gelée
  - groupes de gel
  - fine-tuning tête uniquement
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Les 10 premiers groupes constituent tout le backbone YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Par nom
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Plusieurs sélecteurs
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: "Lister dans l'ordre les groupes de gel d'une famille"
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Geler des éléments

`freeze` est facultatif et ne gèle rien par défaut.

<code-tabs name="train" />

Le gel s'exécute après la construction du modèle et toute reconstruction de la
tête pour un nouveau nombre de classes, mais avant la création de l'optimiseur.
L'optimiseur ne reçoit donc que les paramètres entraînables.

## Formes possibles d'un sélecteur

| Valeur | Signification |
|---|---|
| `None`, `False`, `""`, `"none"` | Entraîner chaque paramètre |
| `10` ou `"10"` | Geler les dix premiers groupes de gel de la famille |
| `[0, 3, 7]` | Geler ces groupes indexés à partir de zéro |
| `"backbone"` | Geler le groupe, le module ou le préfixe de paramètre correspondant |
| `["backbone", "neck"]` | Geler chaque sélecteur listé |
| `["backbone", 3]` | Les listes mixtes fonctionnent |

Une chaîne est analysée avant d'être interprétée. La CLI et une configuration
YAML acceptent donc les mêmes formes que Python. `freeze="[0, 3, 'head']"` est
analysé comme une liste littérale, `freeze="backbone,neck"` est découpé à la
virgule et une chaîne décimale seule devient un nombre.

`freeze=True` est rejeté car ambigu.

Les sélecteurs de noms correspondent à un nom de groupe de gel, un nom de
module ou un préfixe de nom de paramètre, et les caractères glob `*`, `?` et
`[` fonctionnent. Un préfixe `model.` est traité avec souplesse, si bien que
`backbone` et `model.backbone` correspondent tous deux à la graphie utilisée en
interne par la famille.

## Groupes définis par chaque famille

Un entier cible la liste ordonnée de groupes de gel propre à la famille, pas
une position dans un graphe partagé. Les familles de LibreYOLO ne sont pas
toutes un modèle séquentiel unique indexé par YAML. Un numéro de couche brut
aurait donc une signification différente dans chacune.

YOLOv9 ordonne ses groupes depuis l'entrée : dix étapes de backbone, puis six
étapes de neck, puis la tête. C'est pourquoi `freeze=10` correspond exactement
au backbone. `backbone`, `neck` et `head` sont des sélecteurs de noms stables
qui s'y ajoutent.

Les groupes de RF-DETR sont `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` et `head`. Les noms constituent ici le
meilleur choix, car les composants du transformer ne correspondent pas à un
nombre de couches. `backbone` correspond aux deux groupes du backbone par
préfixe.

Les familles qui ne définissent pas de groupes sémantiques utilisent une valeur
par défaut prudente : chaque enfant direct du modèle qui possède au moins un
paramètre, dans l'ordre de déclaration. La liste est généralement courte, si
bien qu'un grand entier ne trouve pas assez de groupes :

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Pour consulter la véritable liste au lieu de deviner :

<code-tabs name="groups" />

## Échecs explicites

Chaque erreur d'utilisation provoque une exception au lieu d'entraîner autre
chose que ce que vous avez demandé.

Un sélecteur sans correspondance provoque une erreur qui nomme les sélecteurs
concernés :

```text
freeze selector(s) matched no parameters: 'backbon'
```

Un gel qui ne laisserait aucun paramètre entraînable provoque une erreur lors
du gel, puis de nouveau lors de la construction de l'optimiseur :

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

C'est ce que fait `freeze="all"`, puisque `all` correspond à chaque paramètre.

Lorsque le gel réussit, une ligne consigne le résultat :

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Arrêt des mises à jour de la BatchNorm gelée

Un paramètre gelé reste dans un module dont les statistiques courantes
continueraient d'évoluer. Chaque module de style BatchNorm dont les paramètres
appartiennent à l'ensemble gelé passe en mode évaluation, et le trainer
réapplique ce mode après l'appel `model.train()` de chaque époque. Les
statistiques restent ainsi fixes pendant toute l'exécution.

Ce comportement est activé par défaut et garantit que geler un backbone le gèle
réellement.

## Combiner avec LoRA

`freeze` et `lora=True` fonctionnent ensemble. Sur RF-DETR, DEIM et ConvNeXt,
les paramètres des adaptateurs restent entraînables même lorsque leur groupe
parent est gelé, ce qui correspond à la combinaison recherchée : un backbone
gelé avec des adaptateurs qui apprennent par-dessus. Consultez le
[fine-tuning LoRA](/docs/train/lora).

## Portée

Il s'agit d'un gel statique décidé au démarrage. Le dégel planifié et le gel
progressif ne font pas partie de l'interface.

## Pages connexes

- [Hyperparamètres](/docs/train/hyperparameters) pour le reste de `train()`.
- [Distillation](/docs/train/distillation) pour l'autre façon de transférer les
  connaissances d'un grand modèle dans un entraînement.
