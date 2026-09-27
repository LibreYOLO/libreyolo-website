---
title: Citation
seo_title: Citer LibreYOLO et les auteurs upstream
description: >-
  Comment citer LibreYOLO dans un article et citer les auteurs de la famille de
  modèles utilisée. Les deux références appartiennent à la même section
  Méthodes.
lead: "Une citation complète de LibreYOLO comporte deux parties\_: la bibliothèque et les travaux publiés à l'origine de la famille de modèles qui a produit le résultat."
keywords:
  - citer libreyolo
  - libreyolo bibtex
  - libreyolo citation cff
  - citation modèle
  - citation vision par ordinateur
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Citer LibreYOLO

Le dépôt publie ses métadonnées de citation dans
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
et non sous forme de bloc BibTeX. GitHub lit ce fichier et propose un bouton
Cite this repository sur la page du dépôt, qui génère les formats APA et
BibTeX. Récupérez l'entrée à cet endroit plutôt que de la saisir vous-même.

Le fichier complet\u00a0:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

Il ne contient délibérément ni version ni date de publication.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
demande aux mainteneurs de ne jamais modifier la version, la date ou le titre
de `CITATION.cff` ou `.zenodo.json` lors d'une publication. Toutes les
citations convergent ainsi vers un enregistrement unique au lieu d'être
dispersées entre les versions. Indiquez la version utilisée dans votre propre
texte et ne modifiez pas la citation.

## Citer la famille de modèles

LibreYOLO est un portage. Exécuter `LibreRFDETRm.pt` revient à exécuter
RF-DETR, et les personnes qui ont créé RF-DETR sont celles qu'un relecteur
s'attend à voir créditées. Citer uniquement la bibliothèque attribue leur
travail au mauvais projet.

Tout le nécessaire figure sur la page de la famille. La ligne Upstream de
l'en-tête nomme les travaux d'origine et leur organisation, avec des liens vers
l'article et le dépôt source. La section Citation plus bas contient le BibTeX.

Ce BibTeX est reproduit verbatim depuis le propre bloc de citation des auteurs,
généralement la section Citation du README upstream ou un fichier
`CITATION.cff`. Il est affiché avec un lien vers son bloc d'origine afin de
pouvoir le vérifier à la source. Il n'est jamais reconstitué depuis les
métadonnées de l'article. Une entrée recréée à la main peut échouer
silencieusement et lourdement\u00a0: coauteur oublié, mauvais lieu de publication,
mauvais type d'entrée ou année de la prépublication. Les prépublications sont
aussi acceptées, une entrée peut donc être un `@inproceedings` même si la
version lue se trouvait sur arXiv.

Copiez le bloc tel quel. Si votre style bibliographique exige un autre type
d'entrée, convertissez l'entrée au lieu de la ressaisir et conservez l'ordre
original des auteurs.

## Contenu requis dans une section Méthodes

Trois éléments rendent un résultat LibreYOLO reproductible et correctement
attribué\u00a0:

- La bibliothèque, citée depuis `CITATION.cff`, avec la version utilisée. `libreyolo version` l'affiche avec les versions de Python, torch et CUDA utilisées.
- Les travaux upstream, cités depuis la section Citation de la page de la famille.
- Le nom exact du checkpoint, comme `LibreRFDETRm.pt`. Les tailles d'une famille se comportent différemment et plusieurs familles publient sous le même préfixe des checkpoints entraînés sur des datasets différents. Le nom de la famille seul n'identifie donc pas l'exécution.

L'attribution constitue également une condition de licence pour une grande
partie des éléments publiés par LibreYOLO. Apache-2.0 et la famille CC BY
exigent que l'avis accompagne les poids redistribués, obligation distincte de
la citation d'un article. Consultez la page sur les
[licences](/docs/licensing) pour connaître les conditions applicables à chaque
checkpoint.

