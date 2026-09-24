---
title: "Toutes les licences YOLO expliquées : de v1 à YOLO26 (guide 2026)"
description: "La carte complète des licences YOLO en 2026 : de YOLOv1 à YOLOv13, YOLO26, YOLO-World et YOLOE, dépôts d'origine et réécritures permissives, avec liens vers les publications et GitHub, et ce que vous pouvez réellement livrer dans un produit commercial."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9 est-il gratuit pour un usage commercial ?"
    a: "Cela dépend du dépôt que vous utilisez. Le dépôt de la publication (WongKinYiu/yolov9) est sous GPL-3.0. Le même laboratoire publie aussi une implémentation distincte de YOLOv9 et YOLOv7 sous licence MIT, dans MultimediaTechLab/YOLO, écrite après que des utilisateurs commerciaux ont demandé une option permissive. Il s'agit d'une réécriture, pas d'un changement de licence des fichiers GPL. La licence du code est claire ; les poids pré-entraînés ne font l'objet d'aucune mention de licence distincte dans aucun des deux dépôts, traitez donc la provenance des poids comme une question à part."
  - q: "Quelles versions de YOLO sont gratuites pour un usage commercial en code fermé ?"
    a: "Par l'intermédiaire d'au moins un dépôt permissif : YOLOv1 à YOLOv4 (Darknet d'origine, domaine public), YOLOv7 et YOLOv9 (la réécriture MIT du laboratoire lui-même), YOLOX et PP-YOLOE (Apache-2.0), ainsi que les réimplémentations MIT de LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World et YOLOE n'ont aucune implémentation permissive en date de juillet 2026. Vérifiez vous-même le fichier LICENSE actuel avant de vous y fier : les licences changent, et ceci est une information générale, pas un conseil juridique."
  - q: "Quelle est la licence de YOLOv10 ?"
    a: "AGPL-3.0. YOLOv10 est une publication académique de l'université Tsinghua, mais son code est construit sur la base de code d'Ultralytics : il hérite donc de l'AGPL-3.0. Il n'existe aucune réimplémentation permissive."
  - q: "Quelle est la licence de YOLOv12 et YOLOv13 ?"
    a: "Les deux sont sous AGPL-3.0, ce que confirment leurs fichiers LICENSE. Comme YOLOv10, ce sont des publications académiques dont le code de référence est construit sur le dépôt Ultralytics : l'AGPL s'applique donc, quel que soit l'auteur de la publication. Les pages tierces qui présentent YOLOv13 comme étant sous Apache-2.0 se trompent."
  - q: "Quand Ultralytics a-t-il fait passer YOLO sous AGPL-3.0 ?"
    a: "Le 14 avril 2023, et non en 2022 comme on le répète souvent. Le fichier LICENSE de ultralytics/yolov5 n'a été modifié que trois fois, et le commit AGPL (34cf749, PR #11359) est daté du 2023-04-14. La dernière version de 2022, YOLOv5 v7.0, était encore distribuée sous GPL-3.0. Le dépôt ultralytics/ultralytics a changé de licence le même jour. YOLOv8 a donc été lancé en janvier 2023 sous GPL-3.0 : les versions PyPI 8.0.0 à 8.0.76 déclarent GPL-3.0, et la 8.0.80 (16 avril 2023) est la première à déclarer AGPL-3.0."
  - q: "YOLO26 est-il gratuit pour un usage commercial ?"
    a: "Pas pour des produits en code fermé. YOLO26 est distribué sous AGPL-3.0, avec une licence Enterprise payante comme alternative : les mêmes conditions que YOLOv8 et YOLO11."
  - q: "Le YOLO Darknet d'origine est-il vraiment dans le domaine public ?"
    a: "Oui. Le fichier LICENSE de Darknet de Joseph Redmon indique « Darknet is public domain. Do whatever you want with it. » (« Darknet est dans le domaine public. Faites-en ce que vous voulez. »). Le fork YOLOv4 d'AlexeyAB contient le même texte de domaine public. Le piège, c'est que les portages PyTorch réellement utilisés ont leurs propres licences, qui vont d'Apache-2.0 à AGPL-3.0, jusqu'à l'absence totale de licence."
  - q: "Puis-je utiliser commercialement les poids pré-entraînés de YOLO-NAS ?"
    a: "Non. Le code de super-gradients est sous Apache-2.0, mais les poids officiels de YOLO-NAS sont distribués sous une licence distincte qui stipule que vous « may not use the Software for any commercial use, including in connection with any models used in a production environment. » (« ne pouvez faire aucun usage commercial du Logiciel, y compris en lien avec des modèles utilisés dans un environnement de production »)."
  - q: "Les poids d'un réseau de neurones héritent-ils de la licence du code d'entraînement ?"
    a: "La question n'est pas tranchée. Ultralytics affirme que l'AGPL-3.0 couvre « the training code and the models produced by that training code » (« le code d'entraînement et les modèles produits par ce code d'entraînement »), et en tant que titulaire des droits d'auteur, l'éditeur fixe les conditions de ce qu'il publie. La question de savoir si un tribunal considérerait que des poids que vous entraînez vous-même sont une œuvre dérivée du code d'entraînement n'a jamais été portée en justice. Considérez les poids AGPL publiés comme grevés de restrictions, et soyez prudent avant de les charger dans une réimplémentation permissive."
  - q: "Pourquoi différentes implémentations d'un même YOLO peuvent-elles avoir des licences différentes ?"
    a: "Le droit d'auteur protège l'expression, pas les idées (17 U.S.C. 102(b)). Une architecture décrite dans une publication peut être implémentée de façon indépendante et placée sous la licence choisie par son auteur. Ce que vous ne pouvez pas faire, c'est copier des fichiers source GPL ou AGPL et en changer la licence. Notez qu'il s'agit uniquement d'un argument de droit d'auteur : l'implémentation indépendante n'est pas une défense contre les brevets."
---

Chaque année apporte de nouveaux YOLO, et chaque année la question de la licence revient, généralement cinq minutes avant une décision produit. Ce qui prête à confusion, c'est que « YOLO » n'est pas un projet unique. C'est un nom de marque partagé par une vingtaine de familles de modèles d'auteurs différents, et voici le détail que la plupart des guides sur les licences oublient : **la licence appartient à un dépôt, pas à un modèle.** Une même version de YOLO peut exister simultanément dans un dépôt GPL et dans un dépôt MIT, des mêmes auteurs. C'est le cas de YOLOv9, et cela change toute la décision.

Ce guide cartographie le paysage dépôt par dépôt, avec des liens vers chaque publication et chaque base de code, à jour en juillet 2026. Chaque licence ci-dessous a été vérifiée dans le fichier LICENSE réel du dépôt réel, car une part surprenante de ce qui s'écrit sur les licences YOLO (y compris dans les guides les mieux classés sur le sujet) est faux.

Si vous voulez seulement une réponse à la question de l'AGPL d'Ultralytics, nous la traitons en détail dans [YOLO est-il gratuit pour un usage commercial ?](/articles/yolo-commercial-license). Celui-ci est la carte complète.

**Deux mises au point en toute transparence avant de commencer.** Premièrement, nous maintenons LibreYOLO, une bibliothèque sous licence MIT qui concurrence plusieurs des projets ci-dessous, et la dernière section de cet article lui est consacrée. C'est une raison de vérifier notre travail, pas de nous croire sur parole, et c'est pourquoi chaque affirmation sur une licence renvoie ici à la source primaire. Deuxièmement, cet article est une information générale sur des textes de licence publiés et des déclarations publiques, à jour à la date indiquée ci-dessus. Ce n'est pas un conseil juridique et il ne remplace pas un avocat dans votre juridiction. Les licences et les positions des mainteneurs changent. Lisez le fichier LICENSE actuel de tout dépôt avant de décider de livrer.

## Le tableau des licences

« Voie permissive » signifie : un dépôt implémentant ce modèle que vous pouvez utiliser dans un produit commercial en code fermé, sans publier le code de votre application et sans payer de licence.

| Modèle | Année | Code d'origine | Voie permissive | Publication |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domaine public) | L'original lui-même | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domaine public) | L'original lui-même | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domaine public) | L'original ; attention au portage PyTorch sous AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (domaine public) | L'original ; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Aucune | pas de publication |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, même laboratoire) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0 ; GPL-3.0 au lancement, voir plus bas) | Aucune (le portage Keras a été abandonné, voir plus bas) | pas de publication |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, même laboratoire) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Aucune | pas de publication |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | L'original lui-même | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | L'original, depuis PaddleDetection (pas PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (code Apache-2.0, poids non commerciaux) | Le code oui, les poids officiels non | pas de publication |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Aucune | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, au point mort depuis 2024) | Aucune | pas de publication |

Lisez-le colonne par colonne et une chose saute aux yeux : le numéro de version ne vous dit rien sur la licence. Le dépôt, si.

## Même modèle, deux licences : le cas YOLOv9

YOLOv9 est la preuve la plus nette que « quelle est la licence de YOLOvN ? » est la mauvaise question, et la chronologie mérite d'être racontée précisément, car c'est la seule fois dans l'histoire de YOLO où la pression de la communauté a réellement changé l'issue.

- **18 février 2024 :** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) devient public en même temps que la [publication](https://arxiv.org/abs/2402.13616).
- **22 février 2024 :** un utilisateur ouvre l'issue #10 pour demander quelle est la licence. Chien-Yao Wang (WongKinYiu) répond « I think it should be GPL3 » (« je pense que ce devrait être GPL3 ») et ajoute le fichier GPL-3.0 quatre jours plus tard.
- **26 février 2024 :** un second utilisateur ouvre l'[issue #82, « An Apache/MIT rewrite »](https://github.com/WongKinYiu/yolov9/issues/82). Au cours des deux semaines et demie qui suivent, une douzaine d'utilisateurs commerciaux s'y joignent.
- **14 mars 2024 :** Wang écrit dans ce fil : *« Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training. »* (« D'accord, je crée un nouveau dépôt pour la réécriture MIT... Je pense pouvoir gérer l'essentiel de l'implémentation des architectures et des fonctions de perte. Et j'ai besoin de quelqu'un pour m'aider sérieusement sur le dataloader et l'entraînement DDP. »)

Ce dépôt, d'abord nommé `yolov9mit`, est aujourd'hui **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)** : sous licence MIT, avec pour copyright « Kin-Yiu, Wong and Hao-Tang, Tsui », il se décrit comme *« the official implementation of YOLOv7 and YOLOv9, YOLO-RD. »* (« l'implémentation officielle de YOLOv7 et YOLOv9, YOLO-RD »). L'essentiel de la réécriture n'a pas été fait par les volontaires du fil, mais par Hao-Tang Tsui, collègue de laboratoire de Wang, qui a écrit environ 90 % de ses commits.

YOLOv9 est donc à la fois « non livrable » et « livrable » selon le dépôt que vous clonez. Même architecture, même laboratoire, deux licences.

**Avant de miser un produit dessus, trois réserves que le dépôt ne vous dira pas en page d'accueil :**

1. **Il est encore en cours de maturation.** Une [issue ouverte](https://github.com/MultimediaTechLab/YOLO/issues/231) de février 2026 signale que les checkpoints fournis ne reproduisent pas les résultats COCO de la publication et comportent nettement plus de paramètres que ce qu'annonce la publication. L'entraînement de la segmentation et des points clés n'est [toujours pas implémenté](https://github.com/MultimediaTechLab/YOLO/issues/232). Rien n'a été fusionné dans main depuis le tag v1.0 de décembre 2025.
2. **Les poids ne font l'objet d'aucune mention de licence distincte.** La LICENSE MIT du dépôt couvre par convention ses assets de release, et les éléments techniques indiquent des checkpoints entraînés par ce projet plutôt que copiés depuis le dépôt GPL : les fichiers diffèrent en taille et en nombre de paramètres de ceux du dépôt GPL. Mais aucun document n'indique que les fichiers .pt sont sous MIT, et une [demande d'audit formel contre une contamination AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) émanant de la communauté est toujours ouverte. C'est une lacune documentaire : personne n'a affirmé que le code était « vérifié sans contamination », et personne n'a allégué de problème.
3. **C'est une base de code différente**, pas un remplacement direct des scripts du dépôt GPL.

Rien de tout cela ne le rend inutilisable. Cela en fait un projet à évaluer, pas une case à cocher.

## Ère 1 : les années du domaine public (YOLOv1 à YOLOv4)

Joseph Redmon a publié [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (paru sous la forme de l'article YOLO9000) et [YOLOv3](https://arxiv.org/abs/1804.02767) dans son framework Darknet, et le [fichier LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) est célèbre pour ne faire que trois lignes :

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

Il s'agit d'une véritable dédicace au domaine public. [YOLOv4](https://arxiv.org/abs/2004.10934), maintenu par Alexey Bochkovskiy dans un [fork de Darknet](https://github.com/AlexeyAB/darknet), comporte le *même* fichier, et non l'Unlicense que plusieurs guides sur les licences lui attribuent. Le classificateur de GitHub lui-même renonce et affiche « Other », ce qu'il vaut la peine de savoir si vos outils de conformité lisent ce champ : un scanner le signalera comme non identifié plutôt que comme une licence permissive propre, alors même que le texte pourrait difficilement être plus permissif.

Darknet lui-même n'est d'ailleurs pas tout à fait mort. Le dépôt de Redmon est inactif depuis 2022, mais le README d'AlexeyAB renvoie désormais vers [hank-ai/darknet](https://github.com/hank-ai/darknet) comme successeur C/C++ recommandé et activement maintenu.

**Le piège, ce sont les portages.** Presque plus personne n'entraîne dans Darknet en 2026 ; les gens se tournent vers une réimplémentation PyTorch, et celles-ci ont leurs propres licences :

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), le portage YOLOv3 le plus populaire, est sous **AGPL-3.0**. Une architecture du domaine public, redistribuée sous la plus stricte des licences courantes dans ce domaine. La licence que vous obtenez dépend entièrement du portage que vous avez installé avec pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) est sous **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) est sous **Apache-2.0** : une voie permissive pour v4 subsiste donc en PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), du co-auteur de YOLOv4, n'a **aucun fichier LICENSE**. C'est pire que le copyleft : sans licence, aucun droit n'est accordé, et le régime par défaut est « tous droits réservés ».

Vérifiez la licence du dépôt que vous avez cloné, pas le numéro de version de la publication.

## Ère 2 : GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) et [YOLOv9](https://github.com/WongKinYiu/yolov9) ont été publiés sous GPL-3.0.

La GPL-3.0 est une licence copyleft : si vous distribuez une version modifiée, ou si vous combinez du code GPL avec le vôtre en un seul programme, vous devez transmettre le code source correspondant complet de cette œuvre combinée sous GPL-3.0. Deux limites comptent. La **simple agrégation** signifie que des programmes séparés et indépendants simplement livrés ensemble ne sont pas automatiquement combinés. Et le déclencheur est la **distribution** : contrairement à l'AGPL, si vous ne livrez jamais l'œuvre combinée, la GPL simple n'impose aucune obligation de divulgation, ce qui explique pourquoi certaines entreprises gardent un modèle GPL strictement derrière leur propre API.

Cette voie est plus étroite qu'il n'y paraît, et elle échoue de trois façons rarement anticipées. Dès que le modèle arrive sur l'appareil d'un client ou dans une installation sur site, vous distribuez. « Usage interne uniquement » cesse d'être vrai dès qu'une copie quitte votre entité juridique : remettre un build à un prestataire, à une équipe de développement externalisée ou à une société affiliée juridiquement distincte est une distribution, même si le copier à l'intérieur d'une même entreprise n'en est pas une. Et le bouclier ne tient que si *tout* ce qui est accessible depuis cette API est sous GPL ou sous une licence plus permissive, car un seul composant AGPL n'importe où dans l'œuvre servie fait basculer l'ensemble sous la section 13, quoi que dise le fichier GPL lui-même.

**YOLOv7 et YOLOv9 disposent de la porte de sortie MIT décrite plus haut.** YOLOv6 non, et la preuve la plus solide vient de Baidu : PaddleDetection est sous Apache-2.0, et ses mainteneurs ont délibérément isolé YOLOv5, YOLOv6, YOLOv7 et YOLOv8 dans un dépôt GPL-3.0 séparé, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), en indiquant que le code de ces modèles « will not be merged into PaddleDetection » (« ne sera pas fusionné dans PaddleDetection »). Quand une entreprise de la taille de Baidu érige un mur pour tenir YOLOv6 à l'écart de son framework permissif, cela vous dit ce que vaut la licence.

## Ère 3 : AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics est passé à l'AGPL-3.0 le **14 avril 2023**, et non en 2022 comme on le répète très largement (y compris dans des guides classés en première page pour cette question). Tout ce qui est sorti depuis ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 et [YOLO26](https://arxiv.org/abs/2606.03748)) est sous AGPL-3.0, avec une licence Enterprise payante comme alternative. Leur [page de licence](https://ultralytics.com/license) indique que le niveau Enterprise couvre « YOLO26, earlier YOLO versions, and any future YOLO models » (« YOLO26, les versions antérieures de YOLO et tout futur modèle YOLO ») : la politique est donc délibérée et tournée vers l'avenir. Les tarifs ne sont pas publics.

Cette date est vérifiable, et mérite d'être vérifiée, car la version populaire de cette histoire est fausse d'une manière qui compte :

- Le fichier `LICENSE` de `ultralytics/yolov5` n'a été modifié que trois fois exactement dans son histoire. La troisième est le commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), « Update LICENSE to AGPL-3.0 (#11359) », daté du **2023-04-14**, qui remplace `GNU GENERAL PUBLIC LICENSE` par `GNU AFFERO GENERAL PUBLIC LICENSE` dans 101 fichiers.
- La dernière version de YOLOv5 de 2022, **v7.0 (22 novembre 2022), était encore distribuée sous GPL-3.0**, tout comme v6.2, v6.1 et v6.0 avant elle. De même pour les milliers de forks qui ont cessé de se synchroniser avant avril 2023 : prenez n'importe lequel d'entre eux, GitHub indique encore GPL-3.0 aujourd'hui.
- Le même jour, 41 minutes plus tard, `ultralytics/ultralytics` a subi [le même traitement](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

D'où le fait que presque personne ne connaît : **YOLOv8 n'a pas été lancé sous AGPL.** Il est sorti en janvier 2023 sous **GPL-3.0** et a changé de licence trois mois plus tard. L'historique PyPI est sans ambiguïté : `ultralytics` 8.0.0 (10 janvier 2023) à 8.0.76 (13 avril 2023) déclarent toutes `GPL-3.0`. La version 8.0.80, mise en ligne le 16 avril 2023, est la première à déclarer `AGPL-3.0`.

Ce n'est pas une échappatoire. Une licence accordée n'est pas révoquée rétroactivement : ces anciennes versions restent donc disponibles aux conditions sous lesquelles elles ont été publiées. Mais elles sont vieilles de trois ans, ni maintenues ni corrigées, la GPL-3.0 reste un copyleft (vous avez échangé une clause réseau contre une clause de distribution, vous n'avez pas échappé au copyleft), et la position commerciale d'Ultralytics va de toute façon au-delà du texte de la licence. La leçon utile est plus étroite et plus générale : **une licence s'attache à la version que vous avez reçue, pas au nom du projet.** Figez vos dépendances et consignez ce que disait la licence le jour où vous avez pris la copie, car le projet peut la changer demain et votre conformité dépend de celle que vous avez réellement obtenue.

L'AGPL-3.0, c'est la GPL plus la section 13, la clause réseau : *si vous modifiez le Programme*, les utilisateurs qui interagissent avec votre version modifiée via un réseau doivent se voir proposer son code source. Cela ferme la voie SaaS que la GPL simple laisse ouverte.

Notez la condition, car la plupart des articles l'omettent. La section 0 de l'AGPL définit « modify » (modifier) comme le fait de copier à partir de l'œuvre ou de l'adapter *d'une manière qui requiert une autorisation au titre du droit d'auteur*. Exécuter un code d'entraînement non modifié sur votre propre dataset, c'est exécuter le programme, pas adapter son code source, de la même façon que compiler votre code avec un compilateur GPL non modifié ne modifie pas le compilateur. L'entraînement seul n'est donc pas automatiquement le déclencheur, et quiconque vous affirme le contraire a sauté la définition.

Ce qui vous fait entrer dans le périmètre de la licence, c'est de combiner ou d'intégrer le package dans votre propre base de code, de sorte que les deux soient livrés ou servis comme un seul programme. Selon la lecture de la FSF elle-même, lier une œuvre GPL ou AGPL à d'autres modules crée une œuvre combinée ; la question de savoir si un tribunal étendrait cela jusqu'à un import Python n'a jamais été portée en justice. Notez aussi que les [conditions commerciales](https://ultralytics.com/license) d'Ultralytics exigent une licence Enterprise pour les plateformes SaaS, les API et les systèmes cloud qui utilisent YOLO en coulisses : la position de l'éditeur va donc au-delà du simple texte de la licence. Considérez l'intégration dans un produit que vous servez ou livrez comme le risque ; ne partez pas du principe que l'entraînement isolé en est un.

Concernant YOLO26 en particulier, les dates sont embrouillées partout, donc : il a été **présenté en avant-première en septembre 2025** lors de YOLO Vision, **réellement publié en janvier 2026** (la version 8.4.0 d'`ultralytics`, dont les notes de version indiquent « Ultralytics YOLO26 has arrived »), et la **publication scientifique est parue en juin 2026**. Il se trouve dans le dépôt principal `ultralytics/ultralytics` ; `ultralytics/yolo26` n'est qu'une page d'accueil.

**Ce qui surprend :** les YOLO académiques des deux dernières années sont eux aussi sous AGPL, et ce n'est pas parce qu'Ultralytics les a écrits.

- [YOLOv10](https://arxiv.org/abs/2405.14458) vient de l'université Tsinghua. Son README indique *« The code base is built with ultralytics. »*
- [YOLOv12](https://arxiv.org/abs/2502.12524) : *« The code is based on ultralytics. »*
- [YOLOv13](https://arxiv.org/abs/2506.17733) : *« The code is based on Ultralytics. »*
- [YOLOE](https://arxiv.org/abs/2503.07465), le modèle à vocabulaire ouvert « see anything » du groupe de YOLOv10, est lui aussi sous AGPL-3.0 et construit sur Ultralytics.

Le copyleft se transmet : une œuvre dérivée de code AGPL doit être transmise sous AGPL, donc dès que l'un de ces forks est livré ou servi, la licence l'accompagne. (Si vous le modifiez et le gardez strictement pour vous, aucune obligation ne s'applique, ce qui explique que l'usage en recherche ne soit pas concerné.) La recherche est indépendante ; la licence, non. Depuis 2024, publier « le prochain YOLO » comme dérivé d'Ultralytics est devenu la méthode par défaut, ce qui signifie que l'AGPL se propage désormais automatiquement au fil des numéros de version.

Si vous voyez une page affirmant que YOLOv13 est sous Apache-2.0, elle se trompe. Le fichier LICENSE, l'API GitHub et la fiche du modèle indiquent tous AGPL-3.0.

### La voie permissive pour YOLOv8 qui n'existe plus

De nombreux guides encore en ligne, y compris une version antérieure de celui-ci, présentent le YOLOv8 Apache-2.0 de KerasCV comme l'échappatoire propre au YOLOv8 sous AGPL. **Vous ne pouvez pas vous y fier aujourd'hui.** Voici ce que montrent les traces publiques :

- Dans la discussion KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), un contributeur a écrit que « many parts are derived form ultralytics » (« de nombreuses parties sont dérivées d'ultralytics »). Aucun mainteneur n'a répondu au fil : la question du degré d'indépendance de l'implémentation n'a donc jamais été tranchée publiquement, dans un sens ou dans l'autre.
- L'issue KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) a été fermée en juillet 2025 par un mainteneur de Keras avec ce message : « Hey all!! Because of license issues, we have decided to drop this. » (« Bonjour à tous ! En raison de problèmes de licence, nous avons décidé d'abandonner ceci. »). Deux personnes ont ensuite demandé ce que cela signifiait pour les utilisateurs commerciaux existants de la version KerasCV. Aucune n'a obtenu de réponse.
- KerasCV est désormais archivé, et KerasHub, son successeur maintenu, **ne fournit pas YOLOv8**.

Nous ne savons pas si ces deux faits sont liés, et nous n'affirmons pas que quoi que ce soit d'irrégulier se soit produit. Ce n'est pas une base sur laquelle bâtir un produit.

## Les exceptions permissives : YOLOX, PP-YOLOE, YOLO-NAS

Tous les YOLO n'ont pas pris le train du copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) est sous Apache-2.0 sans exception, code et poids. C'est toujours le YOLO classique le plus propre pour un usage commercial, même si le dépôt n'a reçu aucun commit depuis la mi-2025 et qu'une [question sur l'intégration des poids pré-entraînés dans une application commerciale](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) est ouverte depuis mars 2026 sans réponse. Le texte de la licence est sans ambiguïté. Il n'y a simplement plus personne en ce moment pour répondre aux questions à son sujet. Nous avons écrit sur [l'utilisation de YOLOX avec LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) est sous Apache-2.0 **au sein de PaddleDetection**. Prenez-le là, et non dans PaddleYOLO, qui contient un répertoire de configuration quasi identique et qui est sous GPL-3.0. Même modèle, même entreprise, deux dépôts, deux licences. C'est la situation de YOLOv9 à l'envers.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) est celui qui piège les gens. Le code est sous Apache-2.0 ; les poids officiels, non. Leur [licence distincte](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) stipule que vous *« may not use the Software for any commercial use, including in connection with any models used in a production environment. »* (« ne pouvez faire aucun usage commercial du Logiciel, y compris en lien avec des modèles utilisés dans un environnement de production »). Depuis le rachat de Deci par NVIDIA en 2024, nous n'avons trouvé aucun nouveau développement fonctionnel dans le dépôt, et les URL d'origine des poids ne répondent plus (les checkpoints sont désormais hébergés ailleurs). Un contournement parfois suggéré, selon lequel entraîner l'architecture à partir de zéro vous laisserait soumis à la seule licence Apache, est plausible mais n'a été confirmé ni par Deci ni par NVIDIA, et il cadre mal avec un texte de licence qui s'étend à « any components comprising the model » (« tout composant constituant le modèle »). Plus d'informations sur YOLO-NAS [ici](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** est parfois présenté comme Apache-2.0 dans d'anciens guides. Ce n'est pas le cas : MMYOLO est sous GPL-3.0 (c'est son projet frère MMDetection qui est sous Apache), et il n'a reçu aucun commit depuis juillet 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) est sous **GPL-3.0**, ni AGPL ni permissif. Cela mérite d'être précisé, car les YOLO à vocabulaire ouvert sont souvent supposés permissifs par association avec l'écosystème de type CLIP dont ils s'inspirent.

Ces cas couvrent les trois erreurs de licence les plus courantes dans ce domaine : supposer que les poids suivent la licence du code, supposer que tout ce qui vient d'une même organisation partage une même licence, et déduire la licence d'un modèle de son nom.

## Les licences couvrent le code, pas les idées (mais attention aux brevets)

Un principe constitue le fondement juridique de toutes les voies permissives ci-dessus : **le droit d'auteur protège l'expression, pas les idées.** Ce n'est pas un slogan, c'est un principe de droit solidement établi des deux côtés de l'Atlantique.

Aux États-Unis, c'est le [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), qui exclut de la protection du droit d'auteur toute « idea, procedure, process, system, method of operation, concept, principle, or discovery » (« idée, procédure, processus, système, méthode de fonctionnement, concept, principe ou découverte »), dans la lignée de *Baker v. Selden* (1879). Dans l'UE, dont le droit s'applique à nous et probablement à bon nombre de lecteurs, l'article 1, paragraphe 2, de la [directive sur les programmes d'ordinateur (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) dispose que les « ideas and principles which underlie any element of a computer program » (« idées et principes qui sont à la base de quelque élément que ce soit d'un programme d'ordinateur ») ne sont pas protégés, et la Cour de justice a confirmé dans l'arrêt *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) que la fonctionnalité d'un programme, son langage de programmation et ses formats de données ne constituent pas en eux-mêmes une expression protégée. Seul le code l'est.

Une architecture décrite dans un article arXiv est une idée publiée. N'importe qui peut l'implémenter. Ce que vous ne pouvez pas faire, c'est copier ou adapter les fichiers source GPL/AGPL de quelqu'un d'autre et changer la licence du résultat. Une implémentation indépendante, écrite par des personnes qui travaillent à partir de la publication et non du code source, est une nouvelle œuvre dont l'auteur choisit la licence.

Deux réserves honnêtes que la plupart des éditeurs passent sous silence :

- **L'« indépendance » doit être réelle.** La version défendable de cette démarche est un processus en salle blanche (clean room) : les personnes qui écrivent le code travaillent à partir de la publication et d'une spécification, pas à partir du code source copyleft. Lire attentivement le dépôt GPL puis « le réécrire », ce n'est pas la même chose, et la différence compte si quelqu'un y regarde un jour de près.
- **C'est un argument de droit d'auteur, pas de brevets.** L'invention indépendante n'est *pas* une défense contre un brevet. Une réimplémentation en salle blanche règle la question du droit d'auteur et laisse celle des brevets exactement là où elle était. Aucun brevet largement revendiqué ne pèse aujourd'hui sur les architectures YOLO, mais « personne n'a encore attaqué en justice » n'équivaut pas à « il n'y a aucun motif de poursuite ».

## Les poids sont une deuxième licence

L'erreur la plus coûteuse dans tout ce domaine consiste à auditer le code et à oublier le checkpoint. Trois couches portent des licences, et elles peuvent toutes différer :

**1. Le code.** Tout ce qui précède.

**2. Les poids.** Parfois de façon explicite, comme pour YOLO-NAS. Parfois par simple affirmation : la [page de licence](https://ultralytics.com/license) d'Ultralytics indique que l'AGPL-3.0 « covers the training code and the models produced by that training code » (« couvre le code d'entraînement et les modèles produits par ce code d'entraînement »), et étend cela aux modèles que vous entraînez vous-même avec leur code. En tant que titulaire des droits sur ce qu'il publie, l'éditeur peut fixer les conditions de ses propres checkpoints. La question de savoir si des poids que *vous* produisez sont juridiquement une œuvre dérivée du code d'entraînement est réellement ouverte : il n'existe aucune jurisprudence, et cela va à l'encontre de la règle générale de la FSF elle-même, selon laquelle la sortie d'un programme n'est pas automatiquement couverte par le droit d'auteur du programme. Non tranché ne veut pas dire sans risque, cependant. En pratique, considérez les poids AGPL publiés comme grevés de restrictions.

Le point sensible pour quiconque suit les conseils de cet article : **les dépôts GPL de YOLOv7 et YOLOv9 ne disent absolument rien de leurs poids.** Leurs checkpoints sont des assets de release dans un dépôt GPL-3.0, sans concession de droits distincte. Charger ces fichiers .pt dans une réimplémentation MIT vous donne un code propre et un checkpoint dont personne n'a clarifié le statut. Si la licence est la raison de votre changement, utilisez des poids que le projet permissif a lui-même entraînés, ou entraînez les vôtres.

**3. Les données d'entraînement.** Les *annotations* de COCO sont sous CC BY 4.0 : les étiquettes ne posent donc pas de problème. Les *images*, en revanche, COCO n'a pas qualité pour les concéder sous licence : ce sont des photos Flickr soumises à un patchwork de conditions individuelles, dont le consortium COCO précise explicitement ne pas être propriétaire. La plupart des acteurs du secteur considèrent ce risque comme faible et passent à autre chose, mais « COCO ne pose pas de problème » est une affirmation qui ne concerne que les annotations. Au-delà de COCO, les choses se durcissent vite : de nombreux datasets aériens, médicaux et de conduite sont non commerciaux, et la licence permissive du code d'un modèle ne les blanchit pas.

Auditez les trois couches, à chaque fois.

## Un mot sur MIT et Apache-2.0

Nous publions une bibliothèque sous licence MIT : il serait donc commode de vous dire que MIT est tout simplement la meilleure licence. La comparaison honnête est plus intéressante.

MIT n'est pas sans obligation : vous devez conserver l'avis de copyright et le texte de la licence dans les copies que vous distribuez. C'est une vraie condition, simplement peu coûteuse. Et Apache-2.0 offre quelque chose que MIT n'offre pas : une **concession expresse de licence de brevet** de la part de chaque contributeur, plus une clause de rétorsion qui met fin à la concession de quiconque intente une action en justice au sujet de l'œuvre. MIT ne dit rien des brevets. Pour une entreprise dont la principale crainte est l'exposition aux brevets, on peut soutenir qu'Apache-2.0 est la plus *sûre* des deux, et une grande partie de l'écosystème permissif de détection (YOLOX, RT-DETR, D-FINE, DEIM) est sous Apache-2.0 exactement pour cette raison.

Ce que MIT vous apporte, c'est la simplicité et la compatibilité. Aucune des deux licences ne vous obligera jamais à publier votre code source, et c'est l'axe dont traite réellement cet article. Si quelqu'un vous dit que MIT contre Apache est la décision importante, il cherche à vous vendre quelque chose. La décision importante, c'est permissif contre copyleft.

## Alors, quel YOLO pouvez-vous réellement utiliser ?

- **Produit commercial en code fermé :** YOLOv7 ou YOLOv9 via le [dépôt MIT](https://github.com/MultimediaTechLab/YOLO) du laboratoire, YOLOX ou PP-YOLOE (depuis PaddleDetection) parmi les originaux, ou un framework MIT maintenu comme LibreYOLO si vous préférez ne pas assembler et auditer tout cela vous-même. Évitez tout ce qui est sous GPL ou AGPL, sauf si vous comptez ouvrir le code de toute votre application ou acheter la licence Enterprise.
- **Projet open source :** n'importe lequel, tant que votre licence est compatible. Si votre projet est sous AGPL, la lignée Ultralytics s'y prête naturellement, et la recherche la plus récente (YOLOv13, YOLO26, YOLOE) y arrive en premier.
- **Recherche et benchmarks :** les licences vous contraignent à peine. Utilisez ce qu'a utilisé la publication à laquelle vous vous comparez.
- **Le plan « on s'en occupera plus tard » :** ne fonctionne pas. Se défaire d'une dépendance AGPL une fois votre produit construit signifie tout réentraîner, revalider et réexporter, poids compris. Choisissez le dépôt, et donc la licence, en premier.

Pour une comparaison des frameworks eux-mêmes plutôt que de leurs licences, consultez [Les meilleures alternatives à Ultralytics en 2026](/articles/best-ultralytics-alternatives).

## L'option MIT : ce que fournit LibreYOLO

Transparence : LibreYOLO est notre projet, et sa licence est la raison d'être de cet article.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) est une base de code unique sous licence MIT** qui couvre les détecteurs ci-dessous derrière une seule API, avec l'entraînement et l'export ONNX, TensorRT, OpenVINO et NCNN intégrés. Pas de copyleft, pas de clause réseau, pas de niveau entreprise.

Voici la gamme complète de détection d'objets, avec le projet amont de chaque famille et sa licence, pour que vous voyiez exactement d'où vient chaque voie permissive :

| Dans LibreYOLO | Modèle | Projet amont | Licence amont |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Les YOLO de l'ère Darknet, en PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Domaine public |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (la réécriture du laboratoire lui-même, pas le dépôt GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, plus une variante end-to-end sans NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (la réécriture du laboratoire lui-même, pas le dépôt GPL) |
| `LibreYOLO9P2` | YOLOv9 avec une tête à stride 4 pour les petits objets | Original LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, détection et pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Code Apache-2.0. Les poids amont soumis à restrictions ne sont **pas** redistribués |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR et v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, détection, segmentation, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 pour N/S/M/L. XL/2XL sont sous la Platform Model License de Roboflow, nous ne livrons donc que les tailles Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Code Apache-2.0. Les plus grandes tailles utilisent le backbone DINOv3 de Meta, sous la propre licence non OSI de Meta, qui autorise la redistribution mais **interdit les usages militaires, nucléaires, d'espionnage et liés aux armes**. Ces tailles sont publiées sous `other`, pas sous Apache. Lisez-la avant de livrer quoi que ce soit à double usage |
| `LibreRTMDet` | RTMDet ([sans MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (le projet frère sous Apache, pas mmyolo sous GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Architecture issue de [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoints via le re-portage [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (les deux) |
| `LibreEC` | EdgeCrafter, détection, pose, segmentation | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Détecteur de centroïdes pour matériel de classe microcontrôleur | Original LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, vocabulaire ouvert (inférence) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, vocabulaire ouvert (inférence) | Google, via `transformers` | Apache-2.0 |

Deux points sur lesquels cette colonne est délibérément honnête.

**Aucun code source GPL ou AGPL n'est copié dans la base de code.** Chaque famille ci-dessus est construite à partir d'un projet amont dans le domaine public, sous MIT ou sous Apache-2.0, et c'est la règle qui permet au framework de rester sous MIT. Lorsqu'une architecture existe à la fois dans un projet permissif et dans un projet copyleft, nous partons du projet permissif : nos YOLOv9 et YOLOv7 remontent au dépôt MIT du laboratoire plutôt qu'aux originaux GPL, et notre RTMDet à MMDetection sous Apache plutôt qu'à MMYOLO sous GPL. YOLO-World est absent de la liste, bien qu'il soit régulièrement demandé, parce qu'il est sous GPL-3.0 et qu'il n'existe aucune source permissive d'où le reprendre.

**Permissif ne veut pas dire sans restriction, et nous préférons le dire plutôt que de vous le laisser découvrir lors d'un audit.** Les plus grandes tailles de DEIMv2 héritent des conditions DINOv3 de Meta, y compris l'interdiction des usages militaires, nucléaires, d'espionnage et liés aux armes, ce qui est une vraie contrainte si vous travaillez dans un domaine proche du double usage. Une poignée de checkpoints en preview de recherche sont entraînés sur des datasets non commerciaux (DOTA, VisDrone) et sont étiquetés `cc-by-nc` sur Hugging Face plutôt que livrés discrètement comme s'ils étaient libres d'usage. Et la prudence que nous appliquons aux poids des autres s'applique aussi aux nôtres : nos checkpoints YOLO9 sont convertis à partir des propres checkpoints du dépôt MIT du laboratoire, ils héritent donc de tout ce qui vaut pour ceux-ci, c'est-à-dire, comme indiqué plus haut, « affirmé, mais pas formellement audité ». La licence MIT couvre le code que nous avons écrit. Chaque checkpoint porte les conditions de ce sur quoi et à partir de quoi il a été entraîné, publiées poids par poids.

Au-delà de la détection, la même API couvre la segmentation d'instances et sémantique, l'estimation de pose, les boîtes orientées, la classification (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), l'estimation de profondeur monoculaire, la restauration d'images et l'estimation du regard.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Le même workflow YOLO, sans les devoirs de licence.

Mettez-lui une étoile sur GitHub : [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentation : [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Marques et affiliation : YOLO, YOLOv8, YOLO11, YOLO26 et Ultralytics sont des marques de leurs propriétaires respectifs, dont Ultralytics Inc. LibreYOLO est un projet indépendant, qui n'est ni affilié à Ultralytics ou à toute autre organisation citée dans cet article, ni sponsorisé ou approuvé par elles. Les noms de produits et de dépôts sont utilisés uniquement pour identifier et comparer les logiciels présentés.*

*Cet article est une information générale, pas un conseil juridique, à jour au 11 juillet 2026, et rédigé par l'un des éditeurs de la comparaison. Vérifiez la licence en vigueur avant de livrer.*
