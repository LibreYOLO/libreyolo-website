---
title: Licences
seo_title: 'Licences de LibreYOLO : code et poids'
description: >-
  Le code propre à LibreYOLO est sous licence MIT. Le code amont intégré et les
  checkpoints publiés ont leurs propres licences, dont plusieurs interdisent
  l'usage commercial.
lead: >-
  LibreYOLO réunit trois éléments soumis à des licences distinctes : son propre
  code, le code amont intégré à une famille de modèles et les checkpoints
  pré-entraînés. Ils ne relèvent souvent pas de la même licence.
keywords:
  - licence libreyolo
  - bibliothèque vision par ordinateur MIT
  - poids modèle usage non commercial
  - licence checkpoint modèle
  - détection objets apache 2
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Code propre à LibreYOLO

La bibliothèque est sous licence MIT. Celle-ci couvre l'API Python, la CLI, les
programmes d'entraînement, de validation et d'exportation, les chargeurs de
datasets et les scripts de conversion du répertoire `weights/`. Vous pouvez
l'utiliser dans un produit commercial ou propriétaire. Conservez la mention de
droit d'auteur et le texte de la licence avec toute copie redistribuée, et vos
obligations s'arrêtent là.

Cette autorisation se limite au code. Le fichier
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
l'énonce clairement :

> Ces licences varient et ne sont pas toutes permissives : certains poids
> publiés sont réservés à un usage non commercial ou font l'objet d'autres
> restrictions, et cette licence MIT ne s'étend pas à eux. Choisir un modèle,
> c'est choisir sa licence.

## Code amont, par famille

La plupart des familles sont des portages de travaux de recherche publiés, et
plusieurs intègrent directement du code source amont. Un fichier ainsi intégré
conserve sa mention de droit d'auteur et sa licence d'origine. La licence MIT ne
les remplace pas, et LibreYOLO ne place pas le travail d'autrui sous une autre
licence. Apache-2.0 et BSD-3-Clause sont les deux licences les plus fréquentes.

Apache-2.0 couvre la lignée DETR et une grande partie des travaux sur les
Transformers : DETR de Meta AI (FAIR), Deformable DETR de SenseTime, LW-DETR de
Baidu, OV-DEIM de Leilei Wang et ses coauteurs, l'implémentation de SegFormer
que LibreYOLO porte depuis Hugging Face Transformers, PP-OCRv5 des auteurs de
PaddlePaddle, SwinIR du Computer Vision Lab de l'ETH Zurich et Depth Anything 3
de ByteDance Seed. Cette licence couvre aussi les classificateurs dérivés de
timm par Ross Wightman et les contributeurs de timm, notamment ResNet, DeiT,
EfficientNetV2, MobileNetV4 et Swin. Le nom de leurs modules reproduit celui de
timm afin que ses tenseurs ImageNet se chargent sans modification.

BSD-3-Clause couvre tout ce qui est dérivé de torchvision : Faster R-CNN,
Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN et DeepLabv3.

MIT couvre un groupe plus restreint, notamment NAFNet de Megvii, CenterNet de
Xingyi Zhou et YOLOv7 tel que republié par ses propres auteurs, Kin-Yiu Wong et
Hao-Tang Tsui, chez MultimediaTechLab. Les familles YOLOv1 à YOLOv4
reproduisent des architectures du projet Darknet de Joseph Redmon et, pour
YOLOv4, d'Alexey Bochkovskiy. Darknet appartient au domaine public et n'impose
donc aucune obligation.

Un sous-arbre inclus n'est pas régi par une licence open source. La famille
DEIMv2 contient le code du backbone DINOv3 de Meta Platforms, soumis au DINOv3
License Agreement, une licence personnalisée non reconnue par l'OSI. Toute
redistribution de ce code doit être accompagnée d'une copie de cet accord.
Celui-ci interdit aussi les utilisations liées aux activités soumises à l'ITAR,
aux objectifs militaires ou guerriers, aux industries nucléaires, à
l'espionnage et au développement d'armes. Ces conditions ne s'appliquent qu'à
ce sous-arbre.

Deux fichiers du dépôt donnent une vue complète.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) recense
chaque sous-arbre tiers inclus, avec son chemin, son fichier de licence et sa
source amont.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
répertorie les projets amont dont LibreYOLO est dérivé et reproduit intégralement
le texte de chaque licence.

## Poids, par checkpoint

Aucun fichier de poids pré-entraînés n'est inclus dans le paquet. Les
checkpoints publiés sont hébergés sur Hugging Face par
l'[organisation LibreYOLO](https://huggingface.co/LibreYOLO), et chaque dépôt
contient son propre fichier `LICENSE` ainsi que l'attribution correspondant au
projet d'origine des poids.

Ce dépôt constitue la source faisant autorité pour les conditions applicables,
et non cette page, la page du modèle ou le résumé dans l'arborescence des
sources. Consultez [checkpoints et poids](/docs/weights) pour connaître le
nommage des fichiers et leur lieu de téléchargement.

Les licences diffèrent d'une famille à l'autre, mais aussi entre les fichiers
d'une même famille. Voici deux exemples de ce second cas :

- Les checkpoints COCO de YOLO9 sont sous licence MIT.
  `LibreYOLO9P2s-visdrone.pt`, entraîné sur VisDrone2019-DET, est sous licence
  CC BY-NC-SA 3.0, qui interdit l'usage commercial.
- Les checkpoints de détection RF-DETR sont sous licence Apache-2.0. Les
  checkpoints de boîtes orientées sont sous licence CC BY 4.0, car ils ont été
  affinés sur un dataset Roboflow Universe publié sous CC BY 4.0 et les poids
  conservent l'exigence d'attribution de ce dataset.

L'éventail est plus large encore entre les familles, et plusieurs checkpoints
publiés ne peuvent pas être utilisés dans un produit commercial :

- SegFormer illustre le plus clairement la séparation entre les deux niveaux.
  Son implémentation est un portage sous licence Apache-2.0 du code de Hugging
  Face Transformers. Les checkpoints ADE20K publiés sont convertis depuis la
  version de NVIDIA sous NVIDIA Source Code License. Cette licence autorise la
  redistribution, mais limite l'utilisation à la recherche ou à l'évaluation
  non commerciale et étend cette restriction aux œuvres dérivées. Ces
  checkpoints ne sont pas couverts par les conditions permissives de LibreYOLO.
- Les checkpoints OV-DEIM sont sous licence CC BY-NC 4.0, comme l'a confirmé
  l'auteur amont. Chaque prédiction charge également la tour de texte
  MobileCLIP-B(LT) d'Apple, dont la licence réserve l'usage à la recherche, une
  condition plus stricte que celle du checkpoint lui-même.
- Le code de SenseNova-Vision est sous licence Apache-2.0 et ses poids sous
  licence CC BY-NC 4.0. Le chargeur affiche l'avis d'utilisation non commerciale
  avant chaque téléchargement automatique.

Certaines familles ne possèdent aucun checkpoint hébergé par LibreYOLO, ce que
leur ligne Poids indique. SAM 3 est soumis à autorisation sur Hugging Face selon
la licence SAM personnalisée de Meta et se télécharge directement depuis Meta.
Les ressources publiées par MiDaS sont récupérées depuis les URL officielles et
leur hachage est vérifié, au lieu d'être réhébergées. Dome-DETR renvoie vers la
source amont, car les métadonnées de sa fiche de modèle ne précisent aucune
licence, tandis que son texte revendique Apache-2.0 tout en réservant
l'utilisation à la recherche universitaire. Ces affirmations sont
contradictoires. Les architectures TEED et DexiNed sont sous licence MIT, mais
les checkpoints publiés par leurs auteurs ont été entraînés sur BIPED, dont les
conditions interdisent l'usage commercial. LibreYOLO ne les inclut donc pas et
ne les télécharge pas automatiquement.

Plusieurs checkpoints torchvision n'ont pas de fichier de licence propre.
LibreYOLO les reproduit sous la licence employée par le projet qui les publie,
précise sur chaque fiche de modèle que ce fondement est implicite plutôt
qu'accordé par checkpoint, et reprend l'avertissement de torchvision selon
lequel les conditions des modèles pré-entraînés peuvent découler des données
d'entraînement.

## Trouver les conditions d'un modèle

La page du modèle comporte dans son en-tête une ligne **Licences**, sous la
forme `Code X, poids Y`, qui renvoie à la section Licences de la page. Cette
section indique le travail original et ses auteurs, la licence amont, la source
amont, la licence du code LibreYOLO, les poids et une interprétation des usages
autorisés. Sur la même page, le tableau Checkpoints comporte une colonne
**Licence des poids**, avec une ligne par fichier publié. Les conditions mixtes
d'une famille sont donc présentées fichier par fichier.

Toutes ces informations sont rendues depuis les mêmes données que celles
contrôlées par la bibliothèque. C'est pourquoi cette page ne les reproduit pas
sous forme de tableau. Une matrice de licences saisie à la main devient erronée
au cours d'une seule version, et une erreur ici coûte cher.

Dans l'arborescence des sources, les équivalents sont `NOTICE` pour le code
inclus, `THIRD_PARTY_NOTICES.txt` pour les projets amont et le texte de leurs
licences, et
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
pour un résumé par famille des checkpoints publiés.

Vérifiez ensuite le dépôt Hugging Face du fichier précis que vous vous apprêtez
à télécharger. Il fait autorité et peut changer sans que la page de
documentation soit modifiée.

## Usage commercial

Le code pose rarement problème. MIT, Apache-2.0 et BSD-3-Clause autorisent
toutes l'utilisation commerciale et propriétaire. Chacune exige de conserver
son texte de licence et ses avis d'attribution avec les copies redistribuées.
Apache-2.0 accorde en outre une licence de brevet, et aucune ne pose de
condition sur le code de votre propre application.

Les produits achoppent plutôt sur les checkpoints. Un checkpoint réservé à un
usage non commercial le reste, aussi permissif que soit le code qui l'entoure,
et la conversion du fichier ne change pas ses conditions applicables, comme
l'indique directement `weights/LICENSE_NOTICE.txt`. Un artefact ONNX ou
TensorRT construit à partir d'un checkpoint restreint hérite de cette
restriction.

Lorsqu'une licence étend sa restriction aux œuvres dérivées, comme le fait la
NVIDIA Source Code License, le fine-tuning ne permet pas non plus d'y échapper.
En revanche, vous pouvez entraîner la même architecture depuis zéro sur des
données que vous avez le droit d'utiliser : le code est permissif, le modèle
que vous entraînez vous appartient et les conditions du checkpoint
pré-entraîné n'interviennent jamais. La page SegFormer le précise pour ses
propres poids. Lisez la ligne Interprétation de la page de chaque famille que
vous envisagez de distribuer.

Tranchez la question de la licence au moment de choisir le modèle, pas lors de
sa livraison. Lisez les conditions du fichier effectivement téléchargé, car une
famille peut proposer côte à côte un checkpoint permissif et un autre soumis à
des restrictions.

## Absence de conseil juridique

Cette page décrit les licences concernées. Il s'agit d'une description, pas
d'un conseil juridique, et elle n'apporte aucune garantie. Si la réponse a des
conséquences commerciales, lisez vous-même les licences et consultez votre
propre conseiller.
