---
title: libreyolo profile
seo_title: "référence de la commande libreyolo profile"
description: "Mesurez la vitesse d'entraînement et d'inférence et lisez le résultat : chaque sous-commande de profile, ses arguments et ses valeurs par défaut, et ce que rapporte chaque angle de lecture."
lead: "Un groupe de commandes qui mesure où passe le temps dans une étape d'entraînement ou dans un appel d'inférence, écrit un profil autonome, et relit ce profil sous plusieurs angles."
keywords: [libreyolo profile cli, profiling entraînement yolo, mesurer latence inférence yolo, profiler kernels gpu pytorch, comparer performances libreyolo]
last_verified: "1.5.0"
meta:
  - label: Commande
    value: libreyolo profile
    mono: true
  - label: Sortie
    value: "profile.json et profile_trace.json sous runs/profile"
    mono: true
snippets:
  examples:
    - label: Mesurer l'inférence
      language: bash
      code: |
        # Sans argument source, l'image d'exemple incluse est utilisée.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Lire le verdict
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Comparer deux mesures
      language: bash
      code: |
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project runs/profile/a
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4 --project runs/profile/b

        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
---

## Synopsis

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Ce groupe n'accepte pas d'arguments `key=value`. Ses sous-commandes utilisent
des arguments positionnels et des flags POSIX, donc c'est
`--weights LibreYOLO9t.pt`, et non `weights=LibreYOLO9t.pt`. Lancer
`libreyolo profile` sans sous-commande affiche la liste.

Deux sous-commandes mesurent et écrivent un profil ; les autres en lisent un.
`run` et `infer` produisent tous deux le même `profile.json` autonome, donc
chaque sous-commande de lecture fonctionne sur l'un comme sur l'autre.

## profile run

Lance un court entraînement profilé et écrit un profil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argument | Défaut | Signification |
|---|---|---|
| `data` | | Positionnel. YAML ou nom du dataset, par exemple `coco128`. Obligatoire |
| `--weights` | `LibreYOLO9t.pt` | Fichier ou nom des poids du modèle |
| `--size` | `t` | Variante de taille du modèle |
| `--batch` | `16` | Micro-batch. `-1` ajuste automatiquement environ 70 % de la VRAM |
| `--imgsz` | `640` | Taille des images d'entraînement |
| `--workers` | `8` | Workers du dataloader |
| `--amp` | `true` | Utilise le chemin AMP de la famille. `--no-amp` le désactive |
| `--steps` | `20` | Étapes profilées, c'est-à-dire mesurées |
| `--warmup` | `5` | Étapes de warmup avant la mesure |
| `--repeat` | `1` | Répète N fois pour obtenir une moyenne et un écart-type |
| `--device` | `0` | Périphérique |
| `--project` | `runs/profile` | Racine du répertoire de sortie |
| `--json` | `false` | Sortie JSON sur stdout |

La fenêtre mesurée vaut `--warmup` plus `--steps` itérations. Un dataset trop
petit pour la remplir ne produit aucun profil et la commande se termine avec le
code `3`, en nommant les trois issues : un dataset plus grand, moins d'étapes,
ou un batch plus petit.

Un `--repeat` supérieur à 1 écrit un `runs/profile/profile_repeat.json` agrégé
dont les métriques scalaires sont moyennées sur les essais, tandis que les
listes de kernels proviennent du dernier essai. C'est aussi le prérequis pour un
verdict de significativité dans `compare` : une seule exécution ne peut pas en
fournir un.

## profile infer

Profile le chemin d'inférence et écrit un profil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argument | Défaut | Signification |
|---|---|---|
| `source` | | Positionnel. Image ou répertoire. L'image d'exemple incluse s'il est omis |
| `--weights` | `LibreYOLO9t.pt` | Fichier ou nom des poids du modèle |
| `--size` | `t` | Variante de taille du modèle |
| `--batch` | `1` | Images par passe avant |
| `--imgsz` | `640` | Taille des images d'entrée |
| `--half` | `false` | Passe avant en autocast, CUDA uniquement. `--no-half` le désactive |
| `--amp-dtype` | `float16` | Dtype de l'autocast CUDA : `float16` ou `bfloat16` |
| `--warmup` | `20` | Itérations de warmup avant la mesure |
| `--runs` | `100` | Itérations mesurées |
| `--repeat` | `1` | Répète N fois pour obtenir une moyenne et un écart-type |
| `--conf` | `0.25` | Seuil de confiance, qui change la quantité de travail de la NMS |
| `--iou` | `0.45` | Seuil IoU de la NMS |
| `--max-det` | `300` | Détections max par image, ce qui change la quantité de travail de la NMS |
| `--device` | `0` | Périphérique |
| `--trace` | `true` | Émet une trace Chrome pour détailler kernels et ops. `--no-trace` l'ignore |
| `--project` | `runs/profile` | Racine du répertoire de sortie |
| `--json` | `false` | Sortie JSON sur stdout |

Rapporte la latence aux p50, p90 et p99, le débit en images par seconde, et la
répartition par étape entre prétraitement, passe avant et post-traitement. Les
trois arguments de seuil sont là parce qu'ils font bouger le chiffre du
post-traitement.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un `profile.json` ou un `profile_trace.json`. Obligatoire |
| `--json` | `false` | Sortie JSON sur stdout |

La lecture de haut niveau : temps par étape, débit, utilisation du GPU, part des
Tensor Cores, pic de VRAM, surcoût côté hôte, lancements de kernels par étape,
le verdict de goulet d'étranglement avec sa raison, la répartition des kernels
par catégorie, et les principaux kernels par étape. Sur un profil d'inférence,
elle affiche aussi les percentiles de latence et la répartition par étape.

Un profil pris en situation de thrashing de VRAM est signalé, car l'utilisation
et le débit qui y sont mesurés ne sont pas fiables.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un profil. Obligatoire |
| `field` | | Positionnel. Nom de la métrique. Omettez-le pour lister les métriques disponibles |
| `--json` | `false` | Sortie JSON sur stdout |

Affiche une métrique et rien d'autre, pour les boucles scriptées. Un champ
inconnu se termine avec le code `2` et renvoie vers la forme qui liste les
métriques.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un profil. Obligatoire |
| `--json` | `false` | Sortie JSON sur stdout |

Millisecondes GPU, millisecondes réelles, nombre de kernels et nombre d'ops par
phase : forward, backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un profil. Obligatoire |
| `--top` | `20` | Affiche les N premiers par temps GPU |
| `--category` | | Filtre par sous-chaîne de catégorie : `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Filtre par expression régulière sur le nom du kernel |
| `--tensorcore` | `false` | Uniquement les kernels Tensor Core |
| `--sort` | `time` | `time`, `count` ou `name` |
| `--phase` | | Restreint à une phase : `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Sortie JSON sur stdout |

Le fond de l'analyse : les kernels GPU individuels avec leur part du temps GPU,
les millisecondes par étape, les invocations par étape et la catégorie. Un
`--phase` inconnu se termine avec le code `2` et liste les phases présentes dans
le profil.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un profil. Obligatoire |
| `--top` | `20` | Affiche les N premiers par temps CPU |
| `--phase` | | Restreint à une phase |
| `--json` | `false` | Sortie JSON sur stdout |

La vue du framework plutôt que celle du périphérique : les ops `aten` et
autograd classées par temps CPU, là où apparaît le coût des lancements côté
hôte.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argument | Défaut | Signification |
|---|---|---|
| `before` | | Positionnel. Profil de référence. Obligatoire |
| `after` | | Positionnel. Nouveau profil. Obligatoire |
| `--json` | `false` | Sortie JSON sur stdout |

Compare le débit, les millisecondes par image, l'utilisation du GPU, le surcoût
côté hôte, les lancements de kernels par étape et le verdict de goulet
d'étranglement.

Le verdict de significativité exige que les deux côtés soient mesurés avec un
`--repeat` d'au moins 2. Dans ce cas, une différence compte comme significative
quand elle dépasse deux fois l'erreur type combinée, et la sortie affiche la
comparaison qu'elle a faite. Sans cela, la ligne indique qu'une seule exécution
ne peut pas soutenir ce verdict.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argument | Défaut | Signification |
|---|---|---|
| `trace` | | Positionnel. Chemin vers un profil. Obligatoire |
| `--remove-category` | | Projette le retrait d'une catégorie de kernels : `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Projette le retrait de N lancements de kernels par étape, par exemple un gain de fusion d'ops |
| `--json` | `false` | Sortie JSON sur stdout |

Estime ce qu'un changement rapporterait avant que le changement soit écrit.
L'une des deux options est obligatoire ; n'en fournir aucune se termine avec le
code `2`.

La projection suit le verdict du profil lui-même. En dessous de 80 %
d'utilisation du GPU, elle modélise le gain comme le nombre de lancements
évités multiplié par le coût hôte mesuré par lancement ; au-dessus, comme moins
de travail GPU. Le résultat porte un champ de mise en garde, car le coût par
lancement est une approximation et la seule preuve reste une seconde mesure.

## Exemples

<code-tabs name="examples" />

## Notes

Le profileur mesure et rapporte. Il ne change rien : la boucle pour laquelle il
est fait consiste à lire le verdict, modifier la configuration ou le code,
relancer, puis comparer.

`--device` vaut `0` par défaut, c'est-à-dire le périphérique CUDA 0. Passer
`--device cpu` mesure sur le CPU et produit un profil que les sous-commandes de
lecture acceptent toujours, sans le détail des kernels GPU.

Chaque sous-commande accepte `--json`, et celles de lecture écrivent uniquement
sur stdout, ce qui rend le groupe utilisable depuis un script.

Les codes de sortie sont ici ceux du groupe : `2` pour un fichier qui n'existe
pas ou un argument qui ne se résout pas, `3` quand `run` n'a produit aucun
profil, et `1` quand une trace ne peut pas être analysée.

À voir aussi : [`libreyolo train`](/docs/cli/train), dont les arguments sont ce
qu'un profil d'entraînement sert généralement à régler.
