---
title: Checkpoints et poids
seo_title: Checkpoints et poids LibreYOLO
description: >-
  Comprendre comment LibreYOLO recherche, télécharge et vérifie les poids des
  modèles, où ils sont hébergés, comment travailler sans réseau et ce qui
  garantit le chargement sûr d'un checkpoint.
lead: >-
  Un checkpoint LibreYOLO est un dictionnaire torch.save qui contient un
  dictionnaire d'état et les métadonnées nécessaires à son identification. Cette
  page explique l'origine de ces fichiers, leur destination et leur chargement.
keywords:
  - poids libreyolo
  - checkpoints libreyolo
  - télécharger poids libreyolo
  - libreyolo hors ligne
  - libreyolo hugging face
  - métadonnées checkpoint
last_verified: 1.5.0
meta:
  - label: Hébergement
    value: 'Un dépôt Hugging Face par checkpoint :'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache local
    value: weights/ sous le répertoire de travail
    mono: true
  - label: Schéma de métadonnées
    value: v1.0
snippets:
  load:
    - label: Téléchargement automatique
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Un simple nom de fichier se résout en weights/LibreYOLO9t.pt et y est
        # téléchargé s'il n'est pas déjà présent.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Chemin explicite
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Un chemin qui contient un répertoire est utilisé exactement tel quel
        # et n'est jamais récupéré depuis le réseau.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Lit les métadonnées sans construire de modèle et indique si elles
        # respectent le schéma.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Renvoie une liste de problèmes. Une liste vide indique que le fichier
        respecte la v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Emplacements de recherche d'un checkpoint

Une référence de modèle dépourvue de répertoire, telle que `LibreYOLO9t.pt`,
est résolue par rapport à `weights/` dans le répertoire de travail actuel. Si
`weights/LibreYOLO9t.pt` existe, ce fichier est utilisé. Si un fichier de ce nom
existe directement dans le répertoire de travail, il est utilisé à la place.
Sinon, `weights/LibreYOLO9t.pt` devient la cible du téléchargement.

Une référence qui contient un répertoire, qu'il soit absolu ou relatif, est
interprétée littéralement. Utilisez cette forme lorsque les poids sont stockés
dans un emplacement central et qu'aucun téléchargement ne doit avoir lieu.

<code-tabs name="load" />

## Téléchargement automatique

Lorsque le chemin résolu n'existe pas, LibreYOLO analyse le nom du fichier pour
retrouver la famille, la taille et la tâche, puis demande à la famille
correspondante une URL de téléchargement. La plupart des familles la
construisent depuis l'organisation LibreYOLO sur Hugging Face, où chaque
checkpoint possède son propre dépôt portant le nom du fichier :

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Le suffixe d'une variante de dataset reste intégré au nom du dépôt. Un
checkpoint entraîné sur autre chose que le dataset par défaut de la famille est
donc résolu vers son propre dépôt au lieu d'écraser le checkpoint par défaut.

Le transfert lui-même est défensif, car un fichier de poids tronqué échouerait
plus tard avec un message peu utile. Les téléchargements sont envoyés par flux
vers un fichier `.part`, puis déplacés atomiquement à leur emplacement
définitif seulement une fois terminés. Un processus interrompu ne peut ainsi
jamais laisser un checkpoint à moitié écrit sur le chemin final. Un transfert
interrompu reprend à son décalage en octets à l'aide d'un validateur HTTP et
redémarre de zéro si le serveur indique que l'objet a changé. Les échecs font
l'objet de trois nouvelles tentatives avec un délai exponentiel. Les processus
simultanés qui ciblent le même chemin prennent un fichier de verrouillage. Deux
entraînements lancés ensemble ne téléchargent donc le fichier qu'une fois.
Lorsqu'une famille récupère un fichier depuis un hôte tiers plutôt que depuis
l'organisation LibreYOLO, elle peut fixer une somme de contrôle et refuser le
fichier en cas de différence.

Si `HF_TOKEN` est défini, ou si un jeton est mis en cache sous
`~/.cache/huggingface/token`, il est joint comme jeton Bearer. Il n'est joint
qu'aux URL de `huggingface.co`. Une famille qui télécharge depuis un autre hôte
ne le reçoit donc jamais.

Toutes les familles ne proposent pas le téléchargement automatique. Certaines
ne renvoient volontairement aucune URL, car les poids publiés ne peuvent pas
être redistribués. L'erreur explique alors ce que vous devez fournir. D'autres
affichent un avis de licence avant le début du transfert. Cet avis à l'exécution
signale que les conditions d'un checkpoint sont plus restrictives que celles du
code. Il mérite d'être lu plutôt qu'ignoré.

## Organisation Hugging Face

Les poids publiés se trouvent sur
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), à raison d'un
dépôt par checkpoint. Chaque dépôt indique une licence, qui n'est pas
nécessairement uniforme dans une famille : une famille dont le code est sous
licence MIT peut proposer certains poids soumis à une autre licence. Le dépôt
fait autorité. Chaque page de modèle répertorie les checkpoints publiés de la
famille et leurs licences dans ses sections Checkpoints et Licences.

## Travailler hors ligne

Une fois les fichiers disponibles localement, aucune partie de la bibliothèque
ne nécessite un accès réseau. Deux méthodes sont possibles.

Préremplissez un répertoire `weights/` à côté de l'emplacement d'exécution de la
tâche. Il suffit de télécharger les checkpoints une fois sur une machine
connectée, puis de copier le répertoire. L'étape de résolution décrite plus haut
les trouve et ne consulte jamais le réseau.

Vous pouvez aussi fournir un chemin absolu vers un emplacement partagé. Une
référence qui contient un répertoire est utilisée telle quelle. Un montage en
lecture seule de poids sélectionnés constitue donc une configuration valide. Si
le processus ne peut pas écrire à côté d'un checkpoint qu'il doit convertir, la
conversion utilise un répertoire temporaire privé au lieu d'échouer.

Les datasets suivent une règle distincte : ils sont résolus sous `~/datasets`
ou dans le répertoire désigné par `LIBREYOLO_DATASETS_DIR` lorsque cette
variable est définie.

## Sécurité du chargement

Les checkpoints sont des objets pickle, et l'ouverture d'un pickle peut exécuter
du code arbitraire. LibreYOLO considère tout fichier de poids comme non fiable
et le charge par le parcours `weights_only=True` de PyTorch, qui limite le
désérialiseur aux tenseurs et à un petit ensemble de types sûrs. Cette règle
s'applique au fichier que vous fournissez, pas seulement à ceux téléchargés par
LibreYOLO. Sur une version de PyTorch trop ancienne pour accepter cet argument,
le chargement est refusé plutôt qu'effectué de manière non sûre.

Certains checkpoints d'entraînement amont contiennent des objets rejetés par le
désérialiseur restreint, comme un objet de configuration du framework qui a
servi à les entraîner. LibreYOLO n'a pas besoin de ces métadonnées. Pendant la
conversion, chaque classe bloquée est donc remplacée par un substitut inerte
qui satisfait le désérialiseur sans rien exécuter, et seuls les tenseurs
subsistent dans le fichier converti. Les noms de modules sensibles sont refusés
entièrement au lieu d'être simulés, et la boucle de nouvelles tentatives est
bornée. Un fichier conçu pour introduire une suite infinie de classes bloquées
échoue donc de façon sûre. Consultez la page
[importer des poids existants](/docs/migrate) pour la suite de ce parcours.

## Métadonnées des checkpoints

Un checkpoint LibreYOLO est un dictionnaire dont la clé `model` contient le
dictionnaire d'état PyTorch. Le schéma v1.0 exige neuf clés qui, ensemble,
permettent à la fabrique d'identifier un fichier sans analyser son nom ni
deviner à partir de la forme des tenseurs.

| Clé | Signification |
|---|---|
| `model` | Dictionnaire d'état PyTorch |
| `schema_version` | Version du contrat de métadonnées. La v1.0 emploie la chaîne `1.0` |
| `libreyolo_version` | Version de LibreYOLO ayant produit le fichier |
| `model_family` | Identifiant d'une famille enregistrée, tel que `yolo9` |
| `size` | Variante au sein de cette famille, telle que `t` ou `r18` |
| `task` | Nom d'une tâche canonique |
| `nc` | Nombre de classes positif |
| `names` | Association de chaque indice de classe à son étiquette, couvrant `0` à `nc - 1` |
| `imgsz` | Résolution d'entrée positive |

Les tâches qui présentent une structure supplémentaire l'enregistrent à côté de
ces clés. Les checkpoints de pose ajoutent `num_keypoints` et `keypoint_dim`,
et peuvent ajouter les sigmas OKS propres à chaque point clé. Les checkpoints
OCR intègrent le jeu de caractères CTC complet, ce qui rend le fichier
autonome. Les checkpoints de restauration peuvent enregistrer le type de
dégradation et un facteur de mise à l'échelle. Les checkpoints du programme
d'entraînement ajoutent l'état de reprise, notamment `epoch`, l'état de
l'optimiseur et les poids EMA. Les poids d'inférence publiés ne devraient pas
les contenir.

Un fichier conforme aux neuf clés se charge par le parcours des métadonnées.
Un fichier non conforme est soit converti si une famille reconnaît sa
structure, soit chargé par le parcours de compatibilité avec un avertissement
qui nomme les éléments manquants.

## Inspecter un checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` ne construit jamais de modèle. Il fonctionne donc sur un
fichier dont la famille n'est pas installée et sur un fichier qui vous inspire
des doutes.
