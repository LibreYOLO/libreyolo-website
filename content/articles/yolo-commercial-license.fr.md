---
title: "YOLO est-il gratuit pour un usage commercial ? Licences YOLOv8, YOLO11 et YOLO26"
description: "YOLOv5, YOLOv8, YOLO11 et YOLO26 sont distribués sous AGPL-3.0 : ils ne sont donc pas gratuits pour un usage commercial en code fermé. Voici ce que la licence exige réellement, version par version, et l'alternative sous licence MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "YOLOv8 est-il gratuit pour un usage commercial ?"
    a: "Pas pour des produits en code fermé. YOLOv8 est sous AGPL-3.0 : vous devez donc soit publier le code source de toute votre application sous AGPL-3.0, soit acheter la licence Enterprise payante de l'éditeur. Pour un usage commercial gratuit et sans restriction, utilisez une bibliothèque sous licence MIT comme LibreYOLO."
  - q: "Quelle est la licence de YOLO ?"
    a: "Les versions principales, YOLOv5, YOLOv8, YOLO11 et YOLO26, sont sous AGPL-3.0 par défaut. L'AGPL-3.0 est une licence copyleft forte : elle est open source, mais vous ne pouvez pas l'utiliser librement dans un produit propriétaire."
  - q: "Puis-je utiliser YOLO dans un produit commercial en code fermé ?"
    a: "Pas sous AGPL-3.0 sans vous y conformer, ce qui implique de publier tout votre produit en open source sous AGPL-3.0. Pour garder votre code privé, soit vous achetez la licence Enterprise de l'éditeur, soit vous passez à un framework sous licence permissive comme LibreYOLO (MIT)."
  - q: "Existe-t-il un YOLO sans AGPL ?"
    a: "Oui. LibreYOLO est sous licence MIT : il n'y a donc ni copyleft AGPL ni clause d'usage en réseau. Vous pouvez l'utiliser librement dans des logiciels commerciaux et en code fermé, sans aucuns frais."
  - q: "Combien coûte une licence commerciale YOLO ?"
    a: "La licence Enterprise de l'éditeur est un contrat payant, conclu entreprise par entreprise, dont les tarifs ne sont pas publics. LibreYOLO ne coûte rien : la licence MIT autorise gratuitement l'usage commercial."
  - q: "Faire tourner YOLO uniquement sur mes propres serveurs permet-il d'échapper à l'AGPL ?"
    a: "Non. La section 13 de l'AGPL-3.0 couvre l'interaction via un réseau. Si des utilisateurs accèdent à votre version modifiée via un réseau, vous devez leur proposer le code source correspondant. Le faire tourner uniquement sur vos propres serveurs ne constitue pas une exemption."
---

**Non. Les modèles YOLO populaires, YOLOv5, YOLOv8, YOLO11 et le nouveau YOLO26, ne sont pas gratuits pour un usage commercial en code fermé.** Ils sont distribués sous **AGPL-3.0**, une licence copyleft forte. Si vous construisez un produit sur l'un d'eux, vous devez publier le code source de *toute* votre application sous AGPL-3.0, ou acheter la licence Enterprise payante de l'éditeur. Si aucune de ces deux options ne vous convient, il existe une alternative sous licence permissive : **[LibreYOLO](/) est sous licence MIT et gratuit pour tout usage commercial, sans contrepartie.**

Si vous avez cherché « YOLOv8 gratuit usage commercial », « licence YOLO » ou « YOLO sans AGPL », cette page y répond précisément, version par version.

## Les licences YOLO en un coup d'œil

| Modèle | Licence par défaut | Gratuit pour un usage commercial en **code fermé** ? | Ce que vous devez faire |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | Non | Publier toute votre application en open source, ou acheter une licence Enterprise |
| YOLOv8 | AGPL-3.0 | Non | Publier toute votre application en open source, ou acheter une licence Enterprise |
| YOLO11 | AGPL-3.0 | Non | Publier toute votre application en open source, ou acheter une licence Enterprise |
| YOLO26 | AGPL-3.0 | Non | Publier toute votre application en open source, ou acheter une licence Enterprise |
| **LibreYOLO** | **MIT** | **Oui** | **Rien. Livrez-le dans des produits propriétaires, gratuitement.** |

En une ligne : les modèles YOLO principaux sont open source, mais sous **AGPL-3.0**, une licence copyleft que la plupart des entreprises ne peuvent pas respecter dans un produit propriétaire. LibreYOLO vous offre le même workflow YOLO sous **licence MIT**, qui ne vous impose aucune obligation de ce type.

## Quelle est la licence de YOLO ?

Les modèles YOLO les plus utilisés, **YOLOv5, YOLOv8, YOLO11 et le tout dernier YOLO26**, sont distribués par leur éditeur sous la **GNU Affero General Public License v3.0 (AGPL-3.0)**.

L'AGPL-3.0 relève du *copyleft fort*. Ce n'est pas l'open source permissif du type « faites ce que vous voulez », comme MIT ou Apache-2.0. Elle comporte une obligation précise et d'une portée considérable, et c'est exactement cette obligation qui piège les utilisateurs commerciaux.

## YOLOv8 est-il gratuit pour un usage commercial ?

Pas au sens où la plupart des gens l'entendent. Vous *pouvez* télécharger YOLOv8 et l'utiliser commercialement, mais uniquement si vous respectez l'AGPL-3.0. En pratique, cela signifie :

- Si vous distribuez, livrez ou hébergez un produit qui inclut YOLOv8, vous devez **publier le code source complet de ce produit**, votre application, votre code d'intégration, vos scripts et votre configuration, sous AGPL-3.0.
- Cette obligation se déclenche non seulement lorsque vous livrez un logiciel à des utilisateurs, mais aussi lorsque vous leur permettez d'interagir avec lui **via un réseau**, la clause de la section 13 qui caractérise l'AGPL. Un SaaS ou un backend d'API n'y échappe pas.

Pour un projet personnel ou de la recherche académique, c'est généralement acceptable. Pour un produit commercial propriétaire, publier l'intégralité de votre base de code est presque toujours hors de question. Tout ce qui précède s'applique de la même façon à **YOLO11** et à **YOLO26** : même licence, même obligation.

## Et YOLOv5, YOLO11 et YOLO26 ?

Même constat. Tous sont sous **AGPL-3.0** par défaut. La licence Enterprise de l'éditeur couvre explicitement « YOLO26, earlier YOLO versions, and any future YOLO models » (« YOLO26, les versions antérieures de YOLO et tout futur modèle YOLO »), ce qui montre que la politique de licence est délibérée et cohérente sur toute la famille. Choisir une version plus récente ne vous donne pas une licence plus accommodante.

Pour être complet : tous les modèles dont le nom contient « YOLO » ne sont pas sous AGPL. Certaines variantes communautaires sont sous GPL-3.0 ou Apache-2.0, et les licences varient selon l'auteur. Mais les versions que les gens ont en tête lorsqu'ils tapent « YOLOv8 » ou « YOLO11 » sont sous AGPL-3.0.

## Ce qu'exige réellement l'AGPL-3.0, en termes simples

L'AGPL-3.0 étend la GPL avec un ajout crucial : la **clause d'usage en réseau**. La check-list pratique :

1. **Distribuer le code source.** Si vous transmettez un logiciel qui inclut du code AGPL, vous devez mettre à disposition le *code source correspondant complet* de l'ensemble de l'œuvre combinée, sous AGPL-3.0.
2. **L'usage en réseau compte.** Si des utilisateurs interagissent avec votre version modifiée via un réseau, une application web, une API, un SaaS, vous devez leur proposer ce même code source. Il n'existe pas d'échappatoire du type « nous le faisons tourner uniquement sur nos serveurs ».
3. **Le copyleft est viral.** L'obligation s'étend à *toute* l'œuvre dérivée, pas seulement aux fichiers YOLO. Votre logique métier entre dans le périmètre de l'AGPL.

Ceci est une information générale, pas un conseil juridique : consultez un avocat pour votre cas précis. Mais la conclusion est simple : **l'AGPL-3.0 et le logiciel commercial en code fermé ne font pas bon ménage.**

## L'option licence Enterprise, et son revers

L'éditeur propose une **licence Enterprise** payante qui lève les obligations de l'AGPL et vous permet d'intégrer YOLO dans un produit propriétaire sans rien publier en open source. C'est une voie légitime, et pour certaines entreprises c'est la bonne.

Les revers :

- **Elle coûte de l'argent** : des frais commerciaux récurrents, négociés entreprise par entreprise, à des tarifs non publics.
- **Elle vous rend dépendant des conditions d'un seul éditeur**, qui peuvent changer et qui ne couvrent que le code de cet éditeur.
- **Vous payez une autorisation** d'utiliser une architecture de modèle qui, au fond, relève de la recherche publiée.

Si des frais récurrents et la dépendance à un fournisseur sont acceptables, la licence Enterprise fonctionne. Si vous préférez ne pas payer, ou ne pas porter cette obligation du tout, lisez la suite.

## L'alternative MIT : LibreYOLO

Transparence : LibreYOLO est notre projet. C'est aussi la raison pour laquelle cette page peut se conclure par une réponse nette plutôt que par un haussement d'épaules.

**LibreYOLO est un framework YOLO publié sous la licence permissive MIT.** Ce seul fait change tout pour l'usage commercial :

- **Gratuit pour un usage commercial**, y compris dans des produits propriétaires en code fermé.
- **Pas de copyleft, pas de clause réseau.** Vous n'avez jamais à publier le code de votre application.
- **Pas de frais par poste ni de licence entreprise.** MIT, c'est MIT.
- **Le même workflow que vous connaissez déjà**, donc la migration est simple.

LibreYOLO est un vrai framework, maintenu, pas un simple wrapper : détection d'objets, segmentation, estimation de pose, classification, profondeur et plus encore, sous une seule API familière, avec l'entraînement et l'export ONNX/TensorRT/OpenVINO/NCNN intégrés. La différence n'est pas le workflow. La différence, c'est la licence.

Si vous arrivez ici avec la question plus large « comment quitter le YOLO sous AGPL », nous avons rédigé une comparaison plus complète de l'écosystème dans [Les meilleures alternatives à Ultralytics en 2026](/articles/best-ultralytics-alternatives).

Pour la licence de tous les autres YOLO, des originaux Darknet dans le domaine public jusqu'à YOLOv9, YOLOv10, YOLOv12, YOLOv13 et YOLO26, consultez [Toutes les licences YOLO expliquées](/articles/yolo-licenses-explained).

### Pourquoi la licence MIT compte pour une entreprise

La licence MIT vous permet d'utiliser, de modifier, d'intégrer et de vendre des logiciels construits sur LibreYOLO **sans obligation de divulguer votre code source** et **sans frais**. C'est la licence d'une grande partie de la pile logicielle moderne, précisément parce qu'elle est sûre pour une adoption commerciale. Votre produit vous appartient ; vous ne devez rien.

## Essayer

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Vous conservez l'expérience de développement YOLO : entraîner sur vos données, prédire et exporter pour livrer. Vous laissez tomber l'obligation AGPL et la facture de la licence entreprise.

LibreYOLO est sous licence MIT, fonctionne sous Linux, Mac et Windows, et tourne sur GPU, Apple Silicon et simple CPU sans changer une ligne de code. Une seule API couvre YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, la lignée RT-DETR, la segmentation, l'estimation de pose, la profondeur et plus encore, le tout sous licences permissives.

Mettez-lui une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
