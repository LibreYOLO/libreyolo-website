---
title: Gel des couches
seo_title: Geler des couches pendant l'entraînement dans LibreYOLO
description: >-
  Geler une partie d'un modèle pour l'apprentissage par transfert : nombre
  entier de groupes de gel propres à la famille, liste explicite d'indices ou
  sélecteurs de noms de modules et de paramètres.
lead: >-
  Le gel maintient certains poids fixes pendant l'entraînement du reste du
  modèle. Les sélecteurs ciblent les groupes de gel ordonnés ou les noms de
  modules propres à une famille, pas des numéros de couches bruts issus d'un
  graphe YAML.
keywords:
  - geler couches
  - apprentissage par transfert
  - geler backbone
  - batchnorm gelée
  - groupes de gel
  - fine tuning tête uniquement
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Les 10 premiers groupes constituent tout le backbone de YOLOv9.
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
    - label: Lister dans l'ordre les groupes de gel d'une famille
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

## Geler une partie du modèle

`freeze` est facultatif et ne gèle rien par défaut.

<code-tabs name="train" />

Le gel intervient après la construction du modèle et toute reconstruction de la
tête pour un nouveau nombre de classes, mais avant la création de l'optimiseur.
Celui-ci ne reçoit donc que les paramètres entraînables.

## Formes possibles d'un sélecteur

| Valeur | Signification |
|---|---|
| `None`, `False`, `""`, `"none"` | Entraîner chaque paramètre |
| `10` ou `"10"` | Geler les dix premiers groupes de gel de la famille |
| `[0, 3, 7]` | Geler ces groupes indexés à partir de zéro |
| `"backbone"` | Geler le groupe, le module ou le préfixe de paramètre correspondant |
| `["backbone", "neck"]` | Geler chaque sélecteur de la liste |
| `["backbone", 3]` | Les listes mixtes fonctionnent |

Une chaîne est analysée avant d'être interprétée. La CLI et une configuration
YAML acceptent donc les mêmes formes que Python. `freeze="[0, 3, 'head']"` est
analysé comme une liste littérale, `freeze="backbone,neck"` est divisé sur la
virgule et une chaîne décimale simple devient un nombre.

`freeze=True` est refusé, car il est ambigu.

Les sélecteurs par nom correspondent à un nom de groupe de gel, un nom de
module ou un préfixe de nom de paramètre, et les caractères glob `*`, `?` et
`[` fonctionnent. Un préfixe `model.` initial est traité avec souplesse.
`backbone` et `model.backbone` atteignent donc la forme employée en interne par
la famille.

## Groupes définis par la famille

Un entier cible la propre liste ordonnée des groupes de gel d'une famille, et
non une position dans un graphe partagé. Les familles LibreYOLO ne sont pas
toutes un modèle séquentiel indexé par YAML. Un numéro de couche brut aurait
donc un sens différent pour chacune.

YOLOv9 ordonne ses groupes depuis l'entrée : dix étages de backbone, puis six
étages de neck et enfin la tête. `freeze=10` correspond donc exactement au
backbone. `backbone`, `neck` et `head` sont des sélecteurs par nom stables qui
s'y ajoutent.

Les groupes de RF-DETR sont `backbone.encoder`, `backbone.projector`,
`decoder`, `queries`, `transformer.encoder_output` et `head`. Les noms
constituent ici le meilleur choix, car les composants Transformer ne
correspondent pas à un nombre de couches. `backbone` atteint les deux groupes
de backbone par préfixe.

Les familles qui ne définissent pas de groupes sémantiques reviennent à une
valeur prudente : chaque enfant direct du modèle qui possède au moins un
paramètre, dans l'ordre de déclaration. Cette liste est généralement courte.
Un entier élevé ne trouvera donc pas suffisamment de groupes :

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Pour consulter la véritable liste au lieu de la deviner :

<code-tabs name="groups" />

## Échecs explicites

Chaque erreur possible déclenche une exception au lieu d'entraîner autre chose
que ce que vous avez demandé.

Un sélecteur sans correspondance déclenche une erreur qui nomme les éléments
manqués :

```text
freeze selector(s) matched no parameters: 'backbon'
```

Un gel qui ne laisserait aucun paramètre entraînable déclenche une erreur, à la
fois au moment du gel et lors de la construction de l'optimiseur :

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

C'est ce que produit `freeze="all"`, puisque `all` correspond à chaque
paramètre.

Lorsque le gel réussit, une ligne enregistre ce qui s'est passé :

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Arrêt des mises à jour de BatchNorm gelée

Un paramètre gelé reste dans un module dont les statistiques cumulées
continueraient normalement d'évoluer. Tout module de type BatchNorm dont les
paramètres appartiennent à l'ensemble gelé passe en mode évaluation. Le
programme d'entraînement réapplique ce mode après chaque appel à `model.train()`
d'une époque. Les statistiques restent donc fixes pendant toute l'exécution.

Ce comportement est activé par défaut et permet au gel d'un backbone de le
geler réellement.

## Composition avec LoRA

`freeze` et `lora=True` fonctionnent ensemble. Sur RF-DETR, DEIM et ConvNeXt,
les paramètres des adaptateurs restent entraînables même lorsque leur groupe
parent est gelé. C'est la combinaison recherchée : un backbone gelé sur lequel
des adaptateurs apprennent. Consultez le
[fine-tuning LoRA](/docs/train/lora).

## Périmètre

Il s'agit d'un gel statique décidé au démarrage. Le dégel planifié et le gel
progressif ne font pas partie de l'interface.

## Voir aussi

- [Hyperparamètres](/docs/train/hyperparameters) pour le reste de `train()`.
- [Distillation](/docs/train/distillation) pour l'autre manière de transférer
  les connaissances d'un grand modèle vers un entraînement.
