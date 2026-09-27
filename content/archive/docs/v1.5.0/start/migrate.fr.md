---
title: Importer des poids existants
seo_title: Charger des poids amont dans LibreYOLO
description: >-
  Indiquer à LibreYOLO un checkpoint issu d'un projet amont. La conversion
  automatique le réencapsule au chargement tout en conservant son nombre et ses
  noms de classes.
lead: >-
  LibreYOLO porte ses familles de modèles depuis des projets amont. Leurs
  checkpoints publiés sont donc presque directement chargeables. Il ne leur
  manque que les métadonnées, que la conversion automatique ajoute au
  chargement.
keywords:
  - convertir poids libreyolo
  - charger checkpoint amont
  - migration libreyolo
  - convertir pth libreyolo
  - conversion automatique checkpoints
last_verified: 1.5.0
meta:
  - label: Point d'entrée
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Écrit à côté de la source sous
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Convertisseurs scriptés
    value: weights/ dans le dépôt
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Remplacez ce chemin par celui d'un checkpoint que vous possédez. Une
        # structure amont reconnue est convertie à la volée, écrite à côté de
        # la source, puis chargée.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # Le nombre et les noms de classes proviennent des tenseurs et des
        # métadonnées propres au fichier. Un fine-tuning conserve donc son
        # ensemble d'étiquettes au lieu de celui de COCO.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Vérifier le résultat
      language: bash
      code: |
        # Le fichier converti respecte le même schéma qu'un fichier publié.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Cette page concerne les checkpoints d'autres projets. Si vous migrez votre
propre code depuis une ancienne version de LibreYOLO, consultez la page
[mettre à niveau vers la version 1.5.0](/docs/upgrade).

## Comportement au chargement d'un fichier externe

`LibreYOLO()` charge d'abord tout fichier de poids par le parcours restreint aux
poids. Si le résultat contient toutes les métadonnées LibreYOLO, il est utilisé
directement. Sinon, le fichier est transmis au convertisseur automatique avant
toute autre tentative. Si le chargement restreint échoue complètement, ce qui
arrive lorsqu'un checkpoint contient un objet tiers sérialisé par pickle, le
convertisseur automatique est essayé avec un chargeur qui neutralise ces
objets.

La conversion automatique effectue quatre opérations. Elle extrait le
dictionnaire de tenseurs de la structure employée par le projet amont. Elle
demande à chaque famille enregistrée si elle reconnaît les clés obtenues et
remappe les noms lorsque le nommage amont diffère du portage de LibreYOLO. Elle
encapsule la famille retenue dans un checkpoint conforme au schéma de
métadonnées v1.0, après avoir lu la taille, la tâche et le nombre de classes
directement dans les tenseurs. Enfin, elle écrit le résultat à côté du fichier
source et le charge.

<code-tabs name="convert" />

La conversion n'est pas silencieuse. Le fichier converti est journalisé avec sa
famille, le nom de la source, le nom de sortie et le nombre de classes obtenu.
Le journal d'une exécution indique donc exactement ce qui a été chargé.

## Structures extraites

Les checkpoints amont imbriquent leurs poids dans quelques emplacements
conventionnels. Le convertisseur les essaie dans l'ordre jusqu'à trouver des
tenseurs : un bloc EMA sous `ema.module` ou un bloc `ema` plat, un
`ema_state_dict` dont le préfixe `module.` est supprimé, puis `params_ema`,
`params`, `ema_net`, `net`, `model`, `state_dict` et enfin l'objet lui-même.
En essayer plusieurs plutôt que de s'arrêter au premier empêche qu'un bloc
`ema` contenant seulement des compteurs masque les véritables poids situés en
dessous.

Les préfixes d'encapsulation sont également retirés : `module.` issu de
l'entraînement distribué, `_orig_mod.` issu d'un modèle compilé et
l'imbrication `model.model.` ajoutée par certaines redistributions.

## Données lues et leur provenance

La taille, la tâche et le nombre de classes proviennent des tenseurs et non du
nom de fichier. C'est pourquoi un checkpoint affiné est converti avec son propre
nombre de classes plutôt qu'avec celui par défaut de l'architecture. Les noms
de classes sont extraits des métadonnées du checkpoint lorsqu'ils y figurent,
ou d'un bloc `args` ou `hyper_parameters` s'ils s'y trouvent. Ils sont limités
au nombre de classes détecté afin qu'un fine-tuning ayant conservé l'ensemble
d'étiquettes de base n'emporte pas des indices absents de sa nouvelle tête.

Les tâches denses sont traitées explicitement au lieu de recevoir des
étiquettes inventées. Un checkpoint de profondeur obtient une classe nommée
`depth`, et un checkpoint de restauration une classe nommée `image`. Un
checkpoint de pose doit fournir un nombre de points clés, soit dans ses
tenseurs, soit dans sa famille. Si aucune de ces sources n'en donne, la
conversion est refusée au lieu de produire un fichier incomplet.

RF-DETR possède son propre reconnaisseur, car la détection de la taille
nécessite le checkpoint entier et parce que sa tête comporte 91 sorties, contre
la convention COCO à 80 classes de LibreYOLO. Un checkpoint est normalisé à
80 classes s'il contient exactement 80 noms, s'il déclare un nombre de classes
égal à 80, s'il désigne COCO comme dataset ou s'il ne contient aucune
métadonnée de classe ou de dataset. Un véritable modèle à 90 classes, identifié
par ses noms, un nombre explicite différent de 80 ou l'indication d'un dataset
autre que COCO, est conservé tel quel.

## Emplacement du fichier converti

La sortie est écrite à côté de la source et porte un nom dérivé de celle-ci :

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Un minuscule détecteur YOLOv9 enregistré sous `upstream-checkpoint.pth` devient
donc `upstream-checkpoint-LibreYOLO9t.pt`. Faire dériver son nom de la source et
non de la famille évite que deux fine-tunings d'une même famille et d'une même
taille présents dans un répertoire s'écrasent, ou entrent en conflit avec un
checkpoint officiel. Le fichier est réécrit à chaque chargement afin de ne
jamais devenir obsolète par rapport à sa source. Si le répertoire est en lecture
seule, le fichier converti est placé dans un nouveau répertoire temporaire
privé, dont le journal indique l'emplacement.

Il s'agit dès lors d'un checkpoint LibreYOLO ordinaire : il se charge par le
parcours des métadonnées et `libreyolo metadata` le déclare valide.

## Cas nécessitant une intervention

Deux familles restent en dehors du reconnaisseur générique. La famille du
regard en est entièrement exclue : elle est limitée à l'inférence et les poids
publiés sont soumis à des restrictions de redistribution. RF-DETR est exclu
parce qu'il utilise le reconnaisseur dédié décrit plus haut, qui le prend en
charge à la place.

Les checkpoints PIDNet amont bruts sont refusés avec une erreur qui renvoie vers
`weights/convert_pidnet_weights.py`. Ce script écrit les métadonnées de
segmentation sémantique Cityscapes dont le checkpoint a besoin.

D-FINE et DEIM partagent les mêmes clés d'architecture. Les tenseurs seuls ne
permettent donc pas de les distinguer. Lorsque les deux revendiquent un fichier
et qu'aucune famille apparentée dotée d'un marqueur distinctif n'est en lice, le
nom de fichier tranche : un nom de la forme `dfine_hgnetv2_n_coco.pth` ou
`deim_hgnetv2_n_coco.pth` suffit, tandis qu'un nom qui ne précise rien est
refusé avec cette explication plutôt que deviné. Instancier directement
`LibreDFINE` ou `LibreDEIM` résout également l'ambiguïté.

Lorsque plusieurs familles revendiquent légitimement un fichier, une sous-classe
l'emporte sur la classe de base qu'elle affine. L'ordre du registre départage
les autres cas, car cet ordre encode le degré de spécificité du contrôle de
chaque famille. Le nom de fichier n'est consulté que pour départager D-FINE et
DEIM. Il ne peut donc jamais faire prévaloir une correspondance générale sur
une correspondance précise.

## Convertisseurs scriptés

Le dépôt contient des scripts de conversion propres à chaque famille sous
`weights/`, ainsi que des fonctions partagées pour les opérations récurrentes.
Ils servent pour un fichier refusé par le parcours d'exécution, pour produire un
checkpoint à l'avance plutôt qu'au chargement et pour les familles dont les
métadonnées doivent être fournies plutôt que déduites des tenseurs.

Ces scripts font partie du dépôt, pas du paquet installé. Leur utilisation
nécessite donc de cloner le projet :

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Chaque script écrit un checkpoint conforme au schéma v1.0, soit la même
exigence que celle respectée par la conversion automatique et par les poids
publiés. Consultez [checkpoints et poids](/docs/weights) pour connaître le
contenu de ce schéma.
