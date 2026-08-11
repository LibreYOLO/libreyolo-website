---
title: Combiner des détecteurs
seo_title: Combiner des détecteurs dans LibreYOLO
description: >-
  Exécutez plusieurs détecteurs sur une image et fusionnez leurs bounding boxes
  par weighted boxes fusion ou NMS, y compris avec des modèles dont les listes
  de classes diffèrent.
lead: >-
  LibreEnsemble exécute au moins deux détecteurs sur une même image décodée et
  fusionne leurs bounding boxes dans un objet Results unique. Chaque membre
  conserve ses propres poids, seuils, appareils et listes de classes.
keywords:
  - ensemble modèles détection objets
  - weighted boxes fusion
  - wbf python
  - combiner deux détecteurs
  - fusionner bounding boxes
  - LibreEnsemble
  - ensemble détection python
  - min_votes
last_verified: 1.5.0
verification: >-
  Signatures du constructeur et de l'appel, valeurs par défaut, erreurs de
  validation, unification des espaces de classes, comptage des votes et objet
  Results renvoyé lus dans libreyolo/ensemble/model.py. Algorithmes de fusion et
  arguments lus dans libreyolo/ops/fusion.py. Intention de conception lue dans
  docs/adr/0004-model-ensembling.md. Schémas d'utilisation recoupés avec
  tests/unit/test_ensemble.py et tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: Deux détecteurs fusionnés
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        # Les membres peuvent être des chemins de checkpoints ou des modèles
        déjà chargés.

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        result = ensemble(SAMPLE_IMAGE)

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Poids et nombre minimal de votes
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # par convention, proportionnels à la mAP de validation
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # conserver uniquement les bounding boxes trouvées par les deux membres
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Seuils par membre
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        # Un scalaire s'applique à tous les membres ; une liste est lue membre
        par membre.

        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)

        print(len(result.boxes))
  external:
    - label: Ajouter un détecteur non chargé par LibreYOLO
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Renvoyer (boxes, scores, labels) : xyxy en pixels de l'image d'origine.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Les mêmes sources qu'un modèle unique
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Remplacez clip.mp4 par un fichier vidéo présent sur le disque.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 4f4c54c52b295795
---

## Définition d'un ensemble

`LibreEnsemble` accepte au moins deux détecteurs, exécute chacun d'eux sur la
même image et fusionne leurs bounding boxes dans un objet `Results` unique.
Cette construction intervient au moment de la prédiction\u00a0: rien n'est à
entraîner et les membres restent des modèles indépendants qui peuvent être
validés et exportés séparément.

La détection est la seule tâche prise en charge. Un membre associé à toute
autre tâche lève `ValueError` lors de la construction, en indiquant l'indice
du membre et sa tâche.

Les deux noms sont importés de façon différée, ils ne coûtent donc rien avant
leur utilisation\u00a0:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Construire un ensemble

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` est une séquence d'au moins deux éléments. Une entrée `str` ou
`Path` est chargée par l'intermédiaire de `LibreYOLO()`\u00a0; toute autre entrée
doit être callable et exposer un dictionnaire `names`. Moins de deux éléments
lèvent `ValueError`, tandis que la transmission d'une chaîne seule lève
`TypeError` au lieu d'itérer sur ses caractères.

Par défaut, `weights` vaut `None`, ce qui applique une pondération uniforme.
Les poids fournis doivent être au nombre d'un par membre et strictement
positifs. Un poids nul lève donc une erreur au lieu de retirer silencieusement
un membre. La convention documentée consiste à les définir proportionnellement
à la mAP de validation de chaque membre.

`fusion_iou` vaut `0.55` par défaut. Il s'agit de l'IoU à partir de laquelle
les bounding boxes de différents membres sont regroupées. Ce seuil est
distinct du paramètre `iou` propre à chaque appel, qui correspond au réglage
NMS de chaque membre.

`min_votes` vaut `1` par défaut, un seul membre peut donc faire retenir une
bounding box. Une valeur supérieure conserve uniquement les groupes confirmés
par ce nombre de membres distincts. Il doit s'agir d'un entier positif qui ne
dépasse pas le nombre de membres. Pour chaque classe, il est limité au nombre
de membres qui connaissent réellement cette classe, afin de ne pas supprimer
silencieusement une classe sur laquelle un seul membre a été entraîné.

## Méthodes de fusion

Trois noms sont acceptés, ainsi qu'un callable.

| `fusion` | Comportement |
|---|---|
| `"wbf"` | Weighted boxes fusion séquentielle et fidèle à l'article. Valeur par défaut |
| `"wbf_seeded"` | Weighted boxes fusion en une passe\u00a0; une NMS consciente des classes choisit les graines des groupes |
| `"nms"` | Concatène les bounding boxes de tous les membres, puis applique une NMS consciente des classes |

La weighted boxes fusion calcule la moyenne des coordonnées d'un groupe,
pondérée par la confiance, et produit ainsi une bounding box qu'aucun membre
n'a proposée seul. Les deux variantes pondérées concordent lorsque les groupes
sont sans ambiguïté, mais peuvent différer légèrement sur des chaînes de
groupes qui se chevauchent. `"nms"` choisit un élément survivant au lieu de
calculer une moyenne. Les survivants conservent donc leurs scores d'origine et
les poids influencent uniquement la bounding box gagnante. Puisqu'elle
sélectionne sans former de groupes, cette méthode ne peut pas compter les
votes\u00a0: associer `fusion="nms"` à une valeur de `min_votes` supérieure à `1`
lève `ValueError`.

La weighted boxes fusion remet à l'échelle le score d'un groupe selon la part
du poids total des membres qui l'ont soutenu. Avec deux membres de même poids,
une bounding box trouvée par un seul conserve la moitié de son score\u00a0: `0.9`
devient `0.45`. Une confiance fusionnée peut donc être inférieure au `conf`
utilisé pour chaque membre. Filtrez selon le score fusionné au lieu de supposer
que le seuil des membres reste respecté.

## Membres aux listes de classes différentes

Les membres n'ont pas besoin de partager la même liste de classes. Leurs
espaces d'étiquettes sont réunis par nom et chaque membre reçoit une table de
correspondance qui remappe ses propres identifiants de classe vers l'union.
`ensemble.names` représente cette union et accompagne l'objet `Results`
renvoyé.

Les bounding boxes ne sont fusionnées qu'au sein d'un même nom de classe. Une
classe connue d'un seul membre traverse la fusion telle quelle, sans être
pénalisée\u00a0: la remise à l'échelle du score utilise un dénominateur propre à la
classe, une classe connue seule conserve donc son score.

Un chevauchement partiel consigne un avertissement qui nomme les classes non
partagées par tous les membres. Lisez attentivement cet avertissement\u00a0: un
checkpoint dont les noms de classes sont des valeurs factices comme `class_0`
construit une union disjointe de tous les autres membres, aucune fusion entre
membres ne se produit alors.

Un membre qui renvoie un identifiant de classe absent de son propre `names`
lève `RuntimeError`.

## Détecteurs externes

<code-tabs name="external" />

`ExternalDetector(fn, names)` encapsule tout callable qui accepte une image
PIL et renvoie `(boxes, scores, labels)`, où les bounding boxes sont exprimées
en xyxy dans les pixels de l'image d'origine. Il valide l'arité, la forme des
bounding boxes, la correspondance des longueurs et la présence de chaque
identifiant de classe dans `names`, puis applique lui-même le seuil `conf`.

C'est ainsi qu'un détecteur non chargé par LibreYOLO peut participer à une
fusion.

## Appeler l'ensemble

<code-tabs name="sources" />

La signature d'appel reproduit celle d'un modèle unique et accepte les mêmes
sources\u00a0: images, dossiers, listes, vidéos, capture d'écran, webcams et flux
réseau. Les sources en direct nécessitent `stream=True` pour la même raison
qu'ailleurs.

| Argument | Valeur par défaut | Remarques |
|---|---|---|
| `conf` | `0.25` | Par membre\u00a0; un scalaire est diffusé, sinon une valeur par membre |
| `iou` | `0.45` | Seuil NMS propre à chaque membre, et non seuil de fusion |
| `imgsz` | `None` | Une `list` est lue membre par membre\u00a0; un `int` ou tuple est diffusé |
| `device` | `None` | Un scalaire ou une valeur par membre, afin de placer les membres sur des appareils différents |
| `classes` | `None` | Filtre le résultat fusionné selon les identifiants de classe de l'union |
| `max_det` | `300` | S'applique au résultat fusionné |

Comme une `list` signifie une valeur par membre pour `imgsz`,
`imgsz=[480, 640]` donne 480 au premier membre et 640 au second, tandis que
`imgsz=(480, 640)` représente une seule taille rectangulaire pour tous. Cette
distinction est facile à manquer.

Les membres sont appelés avec un `max_det` d'au moins 300 quelle que soit la
valeur demandée. Chacun peut ainsi produire largement, puis l'ensemble réduit
le résultat une seule fois à la fin.

L'image n'est décodée qu'une fois et le même objet est transmis à chaque
membre. `batch` est accepté pour assurer la parité, mais ignoré\u00a0; les images
sont traitées séquentiellement.

## Résultat renvoyé

Un objet `Results` ordinaire, du même type que celui renvoyé par un modèle
unique, avec `names` défini sur l'espace de classes de l'union. Tout ce qui
figure dans [Travailler avec les résultats](/docs/predict/results) s'applique
sans modification.

La seule différence est `result.speed`, qu'un ensemble renseigne effectivement.
Ses clés sont `member_0`, `member_1` et ainsi de suite, auxquelles s'ajoute
`fusion`, en millisecondes. C'est le seul endroit de la bibliothèque où
`speed` est renseigné.

Les lignes contenant des bounding boxes ou scores non finis sont supprimées
avant la fusion. Lorsque les membres se trouvent sur des appareils différents,
la fusion s'exécute sur l'appareil du premier membre ayant renvoyé un élément.

## Limites d'un ensemble

`val()` et `export()` lèvent tous deux `NotImplementedError` et vous renvoient
vers les membres\u00a0: validez et exportez chacun d'eux séparément. Aucune méthode
`train` n'existe, son appel lève donc `AttributeError`.

La demi-précision n'est pas gérée au niveau de l'ensemble. `half=True` emprunte
le même chemin sans effet accompagné d'un avertissement que partout ailleurs\u00a0;
configurez la précision sur chaque membre.

Il n'existe aucune interface en ligne de commande pour combiner les modèles.
Cette fonctionnalité est une API Python.
