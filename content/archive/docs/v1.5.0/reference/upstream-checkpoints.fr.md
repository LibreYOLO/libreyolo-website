---
title: Checkpoints upstream
seo_title: Charger des checkpoints upstream dans LibreYOLO
description: "Méthode utilisée par la conversion automatique pour transformer un checkpoint upstream publié en checkpoint LibreYOLO v1.0\_: structures extraites, familles qui reconnaissent chaque format et limites."
lead: >-
  Les familles LibreYOLO sont portées depuis des projets upstream dont les
  checkpoints publiés sont presque chargeables, mais ne contiennent aucune
  métadonnée LibreYOLO. La conversion automatique reconnaît ces fichiers, les
  encapsule selon le schéma v1.0 et écrit le résultat à côté de la source.
keywords:
  - conversion automatique libreyolo
  - charger checkpoint upstream
  - convert_upstream_state_dict
  - poids upstream libreyolo
  - conversion checkpoint
last_verified: 1.5.0
verification: "Comportement lu dans libreyolo/models/autoconvert.py et BaseModel.convert_upstream_state_dict\_; mécanismes de reconnaissance par famille vérifiés en lisant la redéfinition convert_upstream_state_dict de chaque famille, le tout en v1.5.0. Règles COCO de RF-DETR lues dans docs/checkpoint_schema.md."
snippets:
  usage:
    - label: Transmettre simplement le fichier à la fabrique
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Un fichier upstream reconnu est converti au chargement et le
        checkpoint

        # converti est écrit à côté.

        # model = LibreYOLO("yolov9-t-converted.pt")


        # Tout checkpoint LibreYOLO se charge sans modification.

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Comportement au chargement

Lorsque `LibreYOLO()` rencontre un fichier `.pt` qui n'est pas déjà un
checkpoint v1.0 complet, il appelle le convertisseur automatique, qui\u00a0:

1. extrait le dictionnaire de tenseurs des structures upstream courantes\u00a0;
2. demande à chaque famille enregistrée si elle reconnaît la structure, en remappant les clés lorsque les noms upstream diffèrent du portage natif\u00a0;
3. encapsule la famille gagnante dans un checkpoint strict avec métadonnées v1.0, en lisant la taille, la tâche et le nombre de classes dans les tenseurs eux-mêmes afin de convertir correctement les checkpoints affinés\u00a0;
4. l'écrit à côté de la source sous le nom `<source>-<Prefix><size>[-task].pt` et renvoie ce chemin, que la fabrique charge normalement.

Aucune intervention n'est demandée à l'appelant. Si aucune famille ne
revendique le fichier, le convertisseur ne renvoie rien et la fabrique indique
qu'elle n'a pas pu le charger.

<code-tabs name="usage" />

## Structures extraites

Le dictionnaire de tenseurs est recherché dans l'ordre de préférence suivant,
en commençant par l'EMA, et chaque candidat est essayé jusqu'à ce que l'un
contienne réellement des tenseurs. Un bloc EMA vide ou limité aux métadonnées
ne masque donc pas des poids valides placés en dessous.

| Clé | Remarque |
|---|---|
| `ema.module` | Wrapper EMA courant |
| `ema` | Anciens wrappers EMA plats qui stockent directement les tenseurs |
| `ema_state_dict` | Le préfixe `module.` des entrées est retiré |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Le fichier lui-même | Un state dict simple |

Chaque candidat est ensuite réduit à ses entrées contenant des tenseurs et
normalisé\u00a0: tout préfixe initial `module.` ou `_orig_mod.` est retiré, et un
dictionnaire dont toutes les clés commencent par `model.model.` perd ce
préfixe.

## Éléments reconnus par chaque famille

La reconnaissance est une méthode de classe propre à chaque famille.
L'implémentation par défaut revendique une structure dont les clés
correspondent déjà au portage natif. Une famille dont les noms de clés upstream
diffèrent la remplace par un remappage et ne renvoie rien pour les structures
qu'elle ne reconnaît pas.

Familles qui fournissent un mécanisme de reconnaissance avec remappage\u00a0:
`centernet`, `deeplabv3`, `deformable_detr`, `dexined`, `moge2`, `picodet`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`,
`yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`.

Familles qui refusent entièrement la conversion automatique\u00a0: `efficientdet`,
`eomt` et `pidnet` ne renvoient rien depuis leur mécanisme de reconnaissance.
Leurs fichiers upstream doivent donc passer par un script de conversion.
`l2cs` est exclu du mécanisme générique, car cette famille est réservée à
l'inférence et ses poids sont soumis à des restrictions de redistribution.

RF-DETR conserve son propre mécanisme de reconnaissance, car il a besoin du
checkpoint entier et non du seul dictionnaire de tenseurs pour détecter la
taille et remapper les classes COCO. Il n'est enregistré que lorsque ses
dépendances facultatives sont installées.

Toutes les autres familles enregistrées utilisent l'implémentation par défaut\u00a0:
elles revendiquent le fichier lorsque leur propre chargeur reconnaît déjà ces
clés.

## Famille gagnante

Plusieurs familles peuvent revendiquer le même fichier. La résolution reproduit
donc les règles de routage de la fabrique.

La revendication d'une sous-classe l'emporte sur celle de sa classe de base.
L'ordre d'enregistrement suit la création des classes. Une famille dérivée est
donc enregistrée après la base qu'elle affine et ses marqueurs positifs ne
doivent pas perdre face au passage plus large de la base.

L'ordre du registre décide ensuite, car il encode la spécificité\u00a0: la première
revendication est la correspondance la plus précise.

Le seul cas que l'ordre du registre ne peut pas départager oppose DEIM à
D-FINE, dont les clés d'architecture sont identiques. Dans ce cas uniquement,
le nom de fichier sert de signal décisif. Un fichier dont le nom ne donne aucun
indice est refusé plutôt que deviné. Le nom de fichier n'est délibérément
consulté nulle part ailleurs. Une revendication générale faussement positive
ne peut donc jamais être favorisée face à une revendication plus précise à
cause du seul nom du fichier.

## Chargement sécurisé

Les fichiers upstream sont chargés par l'unpickler limité aux poids. Certains
checkpoints d'entraînement upstream intègrent des objets de bibliothèque que
cet unpickler refuse. Ces objets sont des métadonnées d'entraînement et non des
poids. Chaque global bloqué est donc réessayé avec une classe inerte de
remplacement qui satisfait l'unpickler sans rien exécuter. Le nom capturé sert
uniquement d'étiquette textuelle, il n'est jamais importé, évalué ni appelé.

Les noms de modules sensibles sont refusés sans condition et ne sont jamais
remplacés\u00a0: `builtins`, `os`, `sys`, `posix`, `nt` et `subprocess`. La boucle
est limitée à 32 tentatives. Un fichier conçu pour introduire une série sans
fin de globals distincts échoue donc de façon sécurisée au lieu de boucler.
Seuls les tenseurs survivent dans le checkpoint converti.

## Emplacement du fichier converti

La sortie est écrite à côté de la source sous le nom
`<source>-<Prefix><size>[-task].pt`. Elle est toujours réécrite et jamais
réutilisée. Les chargements répétés d'une même source restent ainsi à jour,
tout en évitant les collisions avec les poids officiels ou un autre
fine-tuning de la même famille, taille et tâche dans le même répertoire.

Lorsque le répertoire source est en lecture seule, la conversion se rabat sur
un nouveau répertoire temporaire privé créé pour chaque appel, et la ligne de
log indique le chemin utilisé. La conversion n'est abandonnée avec un
avertissement que si cette méthode échoue elle aussi.

## Checkpoints LibreYOLO existants

Un fichier qui porte un marqueur propre à LibreYOLO, `libreyolo_version` ou
`model_family`, appartient au chemin de chargement normal et n'est pas
reconverti. Cette exclusion s'applique uniquement à une revendication par
passage direct, dont l'ensemble de clés n'a pas changé. Une revendication dont
la conversion a modifié les clés prouve qu'il s'agit d'une structure upstream
étrangère et reste acceptée, même sur un fichier marqué.

`schema_version` n'est délibérément pas traité comme marqueur, car d'autres
outils d'entraînement et d'export emploient ce nom générique. `names`, `nc`,
`size`, `task` et `imgsz` ne le sont pas non plus, puisqu'un fine-tuning
upstream peut également les contenir. Un fine-tuning étranger qui ne porte
qu'une clé générique `names` n'est donc pas marqué. Sa revendication aux clés
natives est convertie normalement et déduit le nombre de classes de la tête
tensorielle plutôt que de le charger à tort comme un modèle à 80 classes.

## Métadonnées lues par le convertisseur

Les noms de classes sont lus dans une clé `names` au niveau supérieur, ou dans
`class_names` au sein d'un bloc `args` ou `hyper_parameters`. Une table de noms
indexée par des étiquettes plutôt que par des indices de classe est inutilisable
et remplacée par des valeurs générées. Une liste de noms plus longue que le
nombre de classes détecté est tronquée, car des indices hors plage feraient
échouer le validateur strict et abandonneraient silencieusement la conversion.

Les `args` upstream sont conservés comme métadonnées simples. Toute valeur qui
n'est pas une chaîne, un nombre, un booléen, une liste ou un dictionnaire est
supprimée, afin qu'aucun élément dangereux n'atteigne le fichier enregistré.

## Normalisation COCO de RF-DETR

Les checkpoints RF-DETR upstream exposent une tête de classification à 91
sorties, soit les 90 classes COCO avec l'arrière-plan. La conversion
automatique normalise un RF-DETR COCO selon la convention COCO-80 et applique
le remappage lors du post-traitement.

Un checkpoint est considéré comme COCO s'il contient exactement 80 noms,
déclare un nombre de classes égal à 80, porte un indice de dataset `coco`, ou
ne contient aucune métadonnée de classe ou de dataset. Ce dernier cas est
important\u00a0: un state dict upstream brut est le checkpoint pré-entraîné COCO
canonique et le seul RF-DETR à 91 sorties sans métadonnées distribué.

Un véritable RF-DETR personnalisé à 90 classes est conservé avec 90 classes.
Une liste de noms, un nombre de classes explicite différent de 80 ou un indice
de dataset non COCO permet de l'identifier. Le repli appliqué aux checkpoints
bruts ne se déclenche donc pas pour lui. Les valeurs factices vides sont
ignorées pour déterminer la présence d'un indice de dataset.

## Limites

La conversion automatique reconnaît les structures upstream publiées. Elle ne
réécrit pas une architecture et ne rend pas chargeable un modèle qui n'a pas
été porté. Lorsqu'aucune famille ne revendique un fichier, la réponse est un
script de conversion et non un argument de fabrique. Le dépôt fournit des
scripts `weights/convert_*.py` pour les familles qui en ont besoin, notamment
EoMT, PIDNet et EfficientDet.

La conversion n'invente pas non plus les métadonnées qu'elle ne peut pas lire.
La taille, la tâche et le nombre de classes proviennent des tenseurs. Les noms
proviennent du fichier lorsqu'ils existent, sinon ils sont générés sous la
forme `class_i`.
