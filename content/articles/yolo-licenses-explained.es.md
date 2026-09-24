---
title: "Todas las licencias de YOLO, explicadas: de v1 a YOLO26 (guía 2026)"
description: "El mapa completo de licencias de YOLO en 2026: de YOLOv1 a YOLOv13, YOLO26, YOLO-World y YOLOE, repositorios originales y reescrituras permisivas, con enlaces a los papers y a GitHub, y qué puedes incluir realmente en un producto comercial."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "¿YOLOv9 es gratis para uso comercial?"
    a: "Depende del repositorio que uses. El repositorio del paper (WongKinYiu/yolov9) es GPL-3.0. El mismo laboratorio publica además una implementación aparte de YOLOv9 y YOLOv7 con licencia MIT en MultimediaTechLab/YOLO, escrita después de que usuarios comerciales pidieran una opción permisiva. Es una reescritura, no un cambio de licencia de los archivos GPL. La licencia del código está clara; los pesos preentrenados no llevan ninguna declaración de licencia propia en ninguno de los dos repositorios, así que trata la procedencia de los pesos como una cuestión aparte."
  - q: "¿Qué versiones de YOLO son gratis para uso comercial de código cerrado?"
    a: "A través de al menos un repositorio permisivo: de YOLOv1 a YOLOv4 (Darknet original, dominio público), YOLOv7 y YOLOv9 (la reescritura MIT del propio laboratorio), YOLOX y PP-YOLOE (Apache-2.0), y las reimplementaciones MIT de LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World y YOLOE no tienen ninguna implementación permisiva a julio de 2026. Comprueba tú mismo el archivo LICENSE actual antes de basarte en esto: las licencias cambian, y esto es información general, no asesoramiento legal."
  - q: "¿Bajo qué licencia está YOLOv10?"
    a: "AGPL-3.0. YOLOv10 es una publicación académica de la Universidad Tsinghua, pero su código está construido sobre el código de Ultralytics, así que hereda AGPL-3.0. No existe ninguna reimplementación permisiva."
  - q: "¿Bajo qué licencia están YOLOv12 y YOLOv13?"
    a: "Ambos son AGPL-3.0, confirmado en sus archivos LICENSE. Como YOLOv10, son papers académicos cuyo código de referencia está construido sobre el repositorio de Ultralytics, así que la AGPL se transmite independientemente de quién escribiera el paper. Las páginas de terceros que indican que YOLOv13 es Apache-2.0 se equivocan."
  - q: "¿Cuándo cambió Ultralytics YOLO a AGPL-3.0?"
    a: "El 14 de abril de 2023, no en 2022 como tanto se repite. El archivo LICENSE de ultralytics/yolov5 solo se ha modificado tres veces, y el commit de la AGPL (34cf749, PR #11359) tiene fecha 2023-04-14. La última versión de 2022, YOLOv5 v7.0, todavía se publicó con GPL-3.0. El repositorio ultralytics/ultralytics cambió de licencia ese mismo día. Por tanto, YOLOv8 se lanzó en enero de 2023 bajo GPL-3.0: las versiones de PyPI de la 8.0.0 a la 8.0.76 declaran GPL-3.0, y la 8.0.80 (16 de abril de 2023) es la primera que declara AGPL-3.0."
  - q: "¿YOLO26 es gratis para uso comercial?"
    a: "No para productos de código cerrado. YOLO26 se distribuye bajo AGPL-3.0 con una licencia Enterprise de pago como alternativa, en las mismas condiciones que YOLOv8 y YOLO11."
  - q: "¿El YOLO original de Darknet es realmente de dominio público?"
    a: "Sí. El LICENSE de Darknet de Joseph Redmon dice 'Darknet is public domain. Do whatever you want with it.' ('Darknet es de dominio público. Haz con él lo que quieras.') El fork de YOLOv4 de AlexeyAB lleva el mismo texto de dominio público. El problema es que los ports a PyTorch que la gente usa de verdad llevan sus propias licencias, desde Apache-2.0 hasta AGPL-3.0, o directamente ninguna licencia."
  - q: "¿Puedo usar comercialmente los pesos preentrenados de YOLO-NAS?"
    a: "No. El código de super-gradients es Apache-2.0, pero los pesos oficiales de YOLO-NAS se distribuyen bajo una licencia aparte que dice que el usuario 'may not use the Software for any commercial use, including in connection with any models used in a production environment' (no puede usar el Software para ningún uso comercial, incluso en relación con modelos usados en un entorno de producción)."
  - q: "¿Los pesos de una red neuronal heredan la licencia del código de entrenamiento?"
    a: "No está resuelto. Ultralytics afirma que AGPL-3.0 cubre 'the training code and the models produced by that training code' (el código de entrenamiento y los modelos producidos por ese código de entrenamiento), y como titular de los derechos de autor fija las condiciones de lo que publica. Nunca se ha puesto a prueba si un tribunal aceptaría que los pesos que entrenas tú mismo son una obra derivada del código de entrenamiento. Trata los pesos AGPL publicados como sujetos a restricciones, y ten cuidado al cargarlos en una reimplementación permisiva."
  - q: "¿Por qué distintas implementaciones del mismo YOLO pueden tener licencias diferentes?"
    a: "Los derechos de autor protegen la expresión, no las ideas (17 U.S.C. 102(b)). Una arquitectura descrita en un paper puede implementarse de forma independiente y licenciarse como decida su autor. Lo que no puedes hacer es copiar archivos fuente GPL o AGPL y cambiarles la licencia. Ten en cuenta que este es solo un argumento de derechos de autor: la implementación independiente no es una defensa frente a las patentes."
---

Cada año hay más YOLO, y cada año se vuelve a plantear la pregunta de la licencia, normalmente cinco minutos antes de una decisión de producto. Lo confuso es que "YOLO" no es un único proyecto. Es una marca compartida por una veintena de familias de modelos de distintos autores, y aquí está el detalle que se les escapa a la mayoría de las guías de licencias: **la licencia pertenece a un repositorio, no a un modelo.** La misma versión de YOLO puede existir a la vez como repositorio GPL y como repositorio MIT, de los mismos autores. Es el caso de YOLOv9, y eso cambia toda la decisión.

Esta guía recorre el panorama repositorio por repositorio, con enlaces a cada paper y a cada código, actualizada a julio de 2026. Cada licencia de abajo se comprobó con el archivo LICENSE real del repositorio real, porque una cantidad sorprendente de lo que se escribe sobre las licencias de YOLO (incluso en las guías mejor posicionadas para esta búsqueda) es erróneo.

Si solo quieres respuesta a la cuestión de la AGPL de Ultralytics, la tratamos a fondo en [¿YOLO es gratis para uso comercial?](/articles/yolo-commercial-license). Esta es el mapa completo.

**Dos avisos antes de empezar.** Primero, mantenemos LibreYOLO, una biblioteca con licencia MIT que compite con varios de los proyectos de abajo, y la última sección de este artículo trata sobre ella. Eso es un motivo para comprobar nuestro trabajo, no para creérnoslo sin más, y por eso cada afirmación sobre licencias enlaza a la fuente primaria. Segundo, este artículo es información general sobre textos de licencias publicados y declaraciones públicas, vigente a la fecha indicada arriba. No es asesoramiento legal ni sustituye a un abogado de tu jurisdicción. Las licencias y las posturas de los mantenedores cambian. Lee el archivo LICENSE actual de cualquier repositorio antes de decidir qué incluyes en un producto.

## La tabla de licencias

"Vía permisiva" significa: un repositorio que implementa este modelo y que puedes usar en un producto comercial de código cerrado, sin publicar tu aplicación como código abierto y sin pagar por una licencia.

| Modelo | Año | Código original | Vía permisiva | Paper |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (dominio público) | El propio original | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (dominio público) | El propio original | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (dominio público) | El original; cuidado con el port a PyTorch con AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (dominio público) | El original; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Ninguna | sin paper |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, mismo laboratorio) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 en el lanzamiento, ver más abajo) | Ninguna (el port de Keras se abandonó, ver más abajo) | sin paper |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, mismo laboratorio) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Ninguna | sin paper |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | El propio original | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | El original, desde PaddleDetection (no PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (código Apache-2.0, pesos no comerciales) | El código sí, los pesos oficiales no | sin paper |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Ninguna | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, parado desde 2024) | Ninguna | sin paper |

Léela columna por columna y algo salta a la vista: el número de versión no te dice nada sobre la licencia. El repositorio sí.

## Mismo modelo, dos licencias: el caso de YOLOv9

YOLOv9 es la prueba más clara de que "¿qué licencia tiene YOLOvN?" es la pregunta equivocada, y merece la pena contar la cronología con precisión porque es la única vez en la historia de YOLO en que la presión de la comunidad cambió realmente el resultado.

- **18 de febrero de 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) se hace público junto con el [paper](https://arxiv.org/abs/2402.13616).
- **22 de febrero de 2024:** un usuario abre el issue #10 preguntando cuál es la licencia. Chien-Yao Wang (WongKinYiu) responde "I think it should be GPL3" (creo que debería ser GPL3) y añade el archivo GPL-3.0 cuatro días después.
- **26 de febrero de 2024:** un segundo usuario abre el [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Durante las dos semanas y media siguientes, se suman una docena de usuarios comerciales.
- **14 de marzo de 2024:** Wang publica en ese hilo: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."* (Vale, creo un repositorio nuevo para la reescritura MIT... Creo que puedo encargarme de la mayor parte de la implementación de las arquitecturas y las funciones de pérdida. Y necesito que alguien me ayude mucho con el dataloader y el entrenamiento DDP.)

Ese repositorio, llamado primero `yolov9mit`, es hoy **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: con licencia MIT, copyright "Kin-Yiu, Wong and Hao-Tang, Tsui", y se describe a sí mismo como *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* (la implementación oficial de YOLOv7 y YOLOv9, YOLO-RD). El grueso de la reescritura no lo hicieron los voluntarios del hilo sino Hao-Tang Tsui, compañero de laboratorio de Wang, que ha escrito aproximadamente el 90 por ciento de sus commits.

Así que YOLOv9 es a la vez "no apto para un producto" y "apto para un producto", según el repositorio que clones. Misma arquitectura, mismo laboratorio, dos licencias.

**Antes de apostar un producto por él, tres advertencias que el repositorio no te cuenta en la portada:**

1. **Todavía está madurando.** Un [issue abierto](https://github.com/MultimediaTechLab/YOLO/issues/231) de febrero de 2026 informa de que los checkpoints publicados no reproducen las cifras COCO del paper y tienen bastantes más parámetros de los que indica el paper. El entrenamiento de segmentación y de keypoints [sigue sin implementar](https://github.com/MultimediaTechLab/YOLO/issues/232). No se ha fusionado nada en main desde la etiqueta v1.0 de diciembre de 2025.
2. **Los pesos no llevan ninguna declaración de licencia propia.** Por convención, la LICENSE MIT del repositorio cubre sus release assets, y las pruebas técnicas apuntan a checkpoints entrenados por este proyecto y no copiados del repositorio GPL: los archivos difieren en tamaño y en número de parámetros de los checkpoints del repositorio GPL. Pero ningún documento declara que los archivos .pt sean MIT, y una [petición de la comunidad de una auditoría formal frente a contaminación AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) sigue abierta. Es un hueco en el papeleo: nadie ha afirmado que esté "verificado como limpio" y nadie ha alegado ningún problema.
3. **Es un código distinto**, no un sustituto directo de los scripts del repositorio GPL.

Nada de eso lo hace inutilizable. Lo convierte en un proyecto que hay que evaluar, no en una casilla que marcar.

## Era 1: los años del dominio público (YOLOv1 a YOLOv4)

Joseph Redmon publicó [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (publicado como el paper de YOLO9000) y [YOLOv3](https://arxiv.org/abs/1804.02767) dentro de su framework Darknet, y el [archivo LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) es famoso por tener tres líneas:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

Es una auténtica dedicación al dominio público. [YOLOv4](https://arxiv.org/abs/2004.10934), mantenido por Alexey Bochkovskiy en un [fork de Darknet](https://github.com/AlexeyAB/darknet), lleva el *mismo* archivo, no la Unlicense que le atribuyen varias guías de licencias. El propio clasificador de GitHub se rinde con él y lo marca como "Other", algo que conviene saber si tus herramientas de cumplimiento leen ese campo: un escáner lo marcará como no identificado en lugar de como una licencia permisiva limpia, aunque el texto difícilmente podría ser más permisivo.

Darknet tampoco está del todo muerto. El repositorio de Redmon está inactivo desde 2022, pero el README de AlexeyAB apunta ahora a [hank-ai/darknet](https://github.com/hank-ai/darknet) como sucesor en C/C++ recomendado y con mantenimiento activo.

**La trampa son los ports.** Casi nadie entrena en Darknet en 2026; la gente recurre a una reimplementación en PyTorch, y esas llevan sus propias licencias:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), el port de YOLOv3 más popular, es **AGPL-3.0**. Una arquitectura de dominio público, redistribuida bajo la licencia más estricta de las habituales en este ámbito. La licencia que obtienes depende por completo de qué port instalaste con pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) es **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) es **Apache-2.0**, así que una vía permisiva para v4 sobrevive en PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), del coautor de YOLOv4, **no tiene ningún archivo LICENSE**. Eso es peor que el copyleft: sin licencia no se concede ningún derecho, y por defecto todos los derechos están reservados.

Comprueba la licencia del repositorio que clonaste, no el número de versión del paper.

## Era 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) y [YOLOv9](https://github.com/WongKinYiu/yolov9) se publicaron como GPL-3.0.

GPL-3.0 es copyleft: si distribuyes una versión modificada, o combinas código GPL con el tuyo en un único programa, debes transmitir el código fuente correspondiente completo de esa obra combinada bajo GPL-3.0. Hay dos límites que importan. La **mera agregación** implica que programas separados e independientes que simplemente se distribuyen juntos no quedan combinados automáticamente. Y el desencadenante es la **distribución**: a diferencia de la AGPL, si nunca distribuyes la obra combinada, la GPL a secas no impone ninguna obligación de divulgación, y por eso algunas empresas mantienen un modelo GPL estrictamente detrás de su propia API.

Esa vía es más estrecha de lo que parece, y falla de tres maneras que rara vez se prevén. En el momento en que el modelo llega al dispositivo de un cliente o a una instalación on-premise, estás distribuyendo. "Solo interno" deja de ser cierto cuando una copia sale de tu entidad jurídica: entregar una build a un contratista, a un equipo de desarrollo externo o a una filial constituida por separado es distribución, aunque copiarla dentro de una misma empresa no lo sea. Y el escudo solo aguanta si *todo* lo alcanzable desde esa API es GPL o más permisivo, porque un solo componente AGPL en cualquier parte de la obra servida arrastra todo el conjunto a la sección 13, diga lo que diga el propio archivo GPL.

**YOLOv7 y YOLOv9 tienen la salida MIT descrita arriba.** YOLOv6 no, y la prueba más contundente viene de Baidu: PaddleDetection es Apache-2.0, y sus mantenedores aislaron deliberadamente YOLOv5, YOLOv6, YOLOv7 y YOLOv8 en un repositorio GPL-3.0 aparte, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), indicando que el código de esos modelos "will not be merged into PaddleDetection" (no se fusionará en PaddleDetection). Cuando una empresa tan grande como Baidu levanta un muro para mantener YOLOv6 fuera de su framework permisivo, eso te dice lo que vale la licencia.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics pasó a AGPL-3.0 el **14 de abril de 2023**, no en 2022 como se repite muchísimo (incluso en guías que aparecen en la primera página para esta pregunta). Todo lo que ha venido después ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 y [YOLO26](https://arxiv.org/abs/2606.03748)) ha sido AGPL-3.0, con una licencia Enterprise de pago como alternativa. Su [página de licencias](https://ultralytics.com/license) dice que el nivel Enterprise cubre "YOLO26, earlier YOLO versions, and any future YOLO models" (YOLO26, versiones anteriores de YOLO y cualquier modelo YOLO futuro), así que la política es deliberada y mira al futuro. Los precios no son públicos.

Esa fecha se puede comprobar, y merece la pena hacerlo, porque la versión popular de esta historia es errónea de una forma que importa:

- El archivo `LICENSE` de `ultralytics/yolov5` se ha modificado exactamente tres veces en su historia. La tercera es el commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)", con fecha **2023-04-14**, que sustituye `GNU GENERAL PUBLIC LICENSE` por `GNU AFFERO GENERAL PUBLIC LICENSE` en 101 archivos.
- La última versión de YOLOv5 de 2022, **v7.0 (22 de noviembre de 2022), todavía se publicó con GPL-3.0**, igual que v6.2, v6.1 y v6.0 antes que ella. Y lo mismo ocurre con los miles de forks que dejaron de sincronizarse antes de abril de 2023: elige cualquiera y GitHub sigue indicando GPL-3.0 hoy.
- Ese mismo día, 41 minutos después, `ultralytics/ultralytics` recibió [el mismo tratamiento](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Lo que produce el dato que casi nadie conoce: **YOLOv8 no se lanzó bajo AGPL.** Se publicó en enero de 2023 bajo **GPL-3.0** y cambió de licencia tres meses después. El registro de PyPI es inequívoco: de `ultralytics` 8.0.0 (10 de enero de 2023) a 8.0.76 (13 de abril de 2023), todas declaran `GPL-3.0`. La versión 8.0.80, subida el 16 de abril de 2023, es la primera que declara `AGPL-3.0`.

Esto no es un resquicio legal. Una licencia concedida no se revoca con carácter retroactivo, así que esas versiones antiguas siguen disponibles en las condiciones con las que se publicaron. Pero tienen tres años de antigüedad, están sin mantenimiento ni parches, GPL-3.0 sigue siendo copyleft (has cambiado una cláusula de red por una cláusula de distribución, no has escapado del copyleft), y la postura comercial de Ultralytics es, de todos modos, más amplia que el texto de la licencia. La conclusión útil es más limitada y más general: **una licencia se aplica a la versión que recibiste, no al nombre del proyecto.** Fija tus dependencias y registra lo que decía la licencia el día en que obtuviste la copia, porque el proyecto puede cambiarla mañana y tu argumento de cumplimiento depende de cuál recibiste realmente.

AGPL-3.0 es la GPL más la sección 13, la cláusula de red: *si modificas el Programa*, a los usuarios que interactúan con tu versión modificada a través de una red se les debe ofrecer su código fuente. Eso cierra la vía SaaS que la GPL a secas deja abierta.

Fíjate en la condición, porque la mayoría de los artículos la omiten. La sección 0 de la AGPL define "modificar" como copiar la obra o adaptarla *de una manera que requiera permiso de copyright*. Ejecutar código de entrenamiento sin modificar sobre tu propio dataset es ejecutar el programa, no adaptar su código fuente, del mismo modo que compilar tu código con un compilador GPL sin modificar no modifica el compilador. Así que entrenar, por sí solo, no es automáticamente el desencadenante, y quien te diga que lo es se ha saltado la definición.

Lo que sí te sitúa dentro del ámbito de la licencia es combinar o integrar el paquete en tu propio código de modo que ambos se distribuyan o se sirvan como un solo programa. La interpretación de la propia FSF es que enlazar una obra GPL o AGPL con otros módulos crea una obra combinada; nunca se ha puesto a prueba si un tribunal extendería eso hasta un import de Python. Ten en cuenta también que las propias [condiciones comerciales](https://ultralytics.com/license) de Ultralytics exigen una licencia Enterprise para plataformas SaaS, APIs y sistemas en la nube que usan YOLO internamente, así que la postura del proveedor es más amplia que el texto literal de la licencia. Trata como riesgo la integración en un producto que sirves o distribuyes; no des por hecho que entrenar de forma aislada lo sea.

Sobre YOLO26 en concreto, las fechas se confunden en todas partes, así que: se **presentó en septiembre de 2025** en YOLO Vision, se **publicó realmente en enero de 2026** (la versión 8.4.0 de `ultralytics`, cuyas notas dicen "Ultralytics YOLO26 has arrived"), y el **paper llegó en junio de 2026**. Vive en el repositorio principal `ultralytics/ultralytics`; `ultralytics/yolo26` es solo una página de presentación.

**Lo que sorprende a la gente:** los YOLO académicos de los dos últimos años también son AGPL, y no porque los escribiera Ultralytics.

- [YOLOv10](https://arxiv.org/abs/2405.14458) es de la Universidad Tsinghua. Su README dice *"The code base is built with ultralytics."* (el código está construido con ultralytics).
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."* (el código se basa en ultralytics).
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."* (el código se basa en Ultralytics).
- [YOLOE](https://arxiv.org/abs/2503.07465), el modelo de vocabulario abierto "see anything" del grupo de YOLOv10, también es AGPL-3.0 y está construido sobre Ultralytics.

El copyleft se hereda: una obra derivada de código AGPL tiene que transmitirse bajo AGPL, así que en el momento en que uno de estos forks se distribuye o se sirve, la licencia va con él. (Si lo modificas y te lo guardas estrictamente para ti, no surge ninguna obligación, y por eso el uso en investigación no se ve afectado.) La investigación es independiente; la licencia no. Desde 2024, publicar "el siguiente YOLO" como derivado de Ultralytics se ha convertido en el flujo de trabajo por defecto, lo que significa que la AGPL se propaga ahora automáticamente a lo largo de los números de versión.

Si ves una página que afirma que YOLOv13 es Apache-2.0, se equivoca. El archivo LICENSE, la API de GitHub y la model card dicen AGPL-3.0.

### La vía permisiva para YOLOv8 que ya no existe

Muchas guías que siguen en línea, incluido un borrador anterior de esta, señalan el YOLOv8 Apache-2.0 de KerasCV como la salida limpia del YOLOv8 con AGPL. **Hoy no puedes confiar en ella.** Esto es lo que consta públicamente:

- En la discusión de KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), un colaborador escribió que "many parts are derived form ultralytics" (muchas partes derivan de ultralytics). Ningún mantenedor respondió en el hilo, así que la cuestión de hasta qué punto la implementación era independiente nunca se ha resuelto públicamente en ningún sentido.
- El issue de KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) lo cerró en julio de 2025 un mantenedor de Keras con: "Hey all!! Because of license issues, we have decided to drop this." (¡Hola a todos! Por problemas de licencia, hemos decidido abandonar esto.) Después, dos personas preguntaron qué significaba eso para los usuarios comerciales existentes de la versión de KerasCV. Ninguna obtuvo respuesta.
- KerasCV está ahora archivado, y KerasHub, su sucesor mantenido, **no incluye YOLOv8**.

No sabemos si esas dos cosas están relacionadas, y no afirmamos que ocurriera nada indebido. Eso no es una base sobre la que construir un producto.

## Las excepciones permisivas: YOLOX, PP-YOLOE, YOLO-NAS

No todos los YOLO se subieron al tren del copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) es Apache-2.0 sin excepciones, código y pesos. Sigue siendo el YOLO clásico más limpio para uso comercial, aunque el repositorio no ha tenido commits desde mediados de 2025, y una [pregunta sobre incluir los pesos preentrenados en una app comercial](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) lleva abierta desde marzo de 2026 sin respuesta. El texto de la licencia es inequívoco. Simplemente, ahora mismo no hay nadie para responder preguntas sobre él. Escribimos sobre [cómo ejecutar YOLOX con LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) es Apache-2.0 **dentro de PaddleDetection**. Tómalo de ahí, no de PaddleYOLO, que contiene un directorio de configuración casi idéntico y es GPL-3.0. Mismo modelo, misma empresa, dos repositorios, dos licencias. Es la situación de YOLOv9 al revés.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) es el que pilla a la gente. El código es Apache-2.0; los pesos oficiales no. Su [licencia aparte](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) establece que el usuario *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* (no puede usar el Software para ningún uso comercial, incluso en relación con modelos usados en un entorno de producción). Desde que NVIDIA adquirió Deci en 2024 no hemos encontrado nuevo trabajo de funcionalidades en el repositorio, y las URL originales de los pesos ya no funcionan (los checkpoints están ahora en otro host). Una solución que a veces se sugiere, que entrenar la arquitectura desde cero te deja limpio bajo Apache, es plausible pero no la han confirmado ni Deci ni NVIDIA, y encaja mal con un texto de licencia que alcanza a "any components comprising the model" (cualquier componente que forme parte del modelo). Más sobre YOLO-NAS [aquí](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** aparece a veces como Apache-2.0 en guías antiguas. No lo es: MMYOLO es GPL-3.0 (su hermano MMDetection es el Apache), y no ha tenido commits desde julio de 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) es **GPL-3.0**, ni AGPL ni permisivo. Merece la pena decirlo porque a menudo se supone que los YOLO de vocabulario abierto son permisivos por asociación con el ecosistema de estilo CLIP del que toman prestado.

Esos casos cubren los tres errores de licencias más comunes en este ámbito: suponer que los pesos siguen la licencia del código, suponer que todo lo de una misma organización comparte licencia, y deducir la licencia de un modelo por su nombre.

## Las licencias cubren el código, no las ideas (pero ojo con las patentes)

Un principio es la base legal de todas las vías permisivas anteriores: **los derechos de autor protegen la expresión, no las ideas.** No es un eslogan, es derecho consolidado a ambos lados del Atlántico.

En EE. UU. es el [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), que excluye de la protección de los derechos de autor cualquier "idea, procedure, process, system, method of operation, concept, principle, or discovery" (idea, procedimiento, proceso, sistema, método de funcionamiento, concepto, principio o descubrimiento), con origen en *Baker v. Selden* (1879). En la UE, que es la ley que nos rige a nosotros y probablemente a buena parte de los lectores, el artículo 1(2) de la [Directiva sobre programas de ordenador (2009/24/CE)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) dice que "ideas and principles which underlie any element of a computer program" (las ideas y principios en los que se base cualquiera de los elementos de un programa de ordenador) no están protegidos, y el Tribunal de Justicia confirmó en *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) que la funcionalidad de un programa, su lenguaje de programación y sus formatos de datos no son en sí mismos expresión protegida. Solo el código lo es.

Una arquitectura descrita en un paper de arXiv es una idea publicada. Cualquiera puede implementarla. Lo que no puedes hacer es copiar o adaptar los archivos fuente GPL/AGPL de otra persona y cambiar la licencia del resultado. Una implementación independiente, escrita por personas que trabajan a partir del paper y no del código fuente, es una obra nueva cuyo autor elige la licencia.

Dos advertencias honestas que la mayoría de los proveedores omiten:

- **"Independiente" tiene que ser verdad.** La versión defendible de esto es un proceso de sala limpia (clean room): quienes escriben el código trabajan a partir del paper y de una especificación, no del código fuente copyleft. Leer el repositorio GPL con detalle y luego "reescribirlo" no es lo mismo, y la diferencia importa si alguien llega a examinarlo.
- **Este es un argumento de derechos de autor, no de patentes.** La invención independiente *no* es una defensa frente a una patente. Una reimplementación en sala limpia resuelve la cuestión de los derechos de autor y deja la de las patentes exactamente donde estaba. Hoy no hay ninguna patente ampliamente invocada que pese sobre las arquitecturas YOLO, pero "nadie ha demandado todavía" no es lo mismo que "no hay nada por lo que demandar".

## Los pesos son una segunda licencia

El error más caro en todo este ámbito es auditar el código y olvidarse del checkpoint. Hay tres capas con licencia, y todas pueden ser distintas:

**1. El código.** Todo lo anterior.

**2. Los pesos.** A veces de forma explícita, como con YOLO-NAS. A veces por afirmación: la [página de licencias](https://ultralytics.com/license) de Ultralytics dice que AGPL-3.0 "covers the training code and the models produced by that training code" (cubre el código de entrenamiento y los modelos producidos por ese código de entrenamiento), y lo extiende a los modelos que entrenas tú mismo con su código. Como titular de los derechos de autor de lo que publica, puede fijar las condiciones de sus propios checkpoints. Si los pesos que produces *tú* son legalmente una obra derivada del código de entrenamiento es una cuestión realmente abierta: no hay jurisprudencia, y choca con la regla general de la propia FSF de que la salida de un programa no está cubierta automáticamente por los derechos de autor del programa. Aun así, no resuelto no es lo mismo que seguro. En la práctica, trata los pesos AGPL publicados como sujetos a restricciones.

El punto delicado para quien siga los consejos de este artículo: **los repositorios GPL de YOLOv7 y YOLOv9 no dicen nada sobre sus pesos.** Sus checkpoints son release assets de un repositorio GPL-3.0 sin ninguna concesión aparte. Cargar esos archivos .pt en una reimplementación MIT te da código limpio y un checkpoint cuyo estado nadie ha aclarado. Si la licencia es el motivo por el que cambiaste, usa pesos que el propio proyecto permisivo haya entrenado, o entrena los tuyos.

**3. Los datos de entrenamiento.** Las *anotaciones* de COCO son CC BY 4.0, así que las etiquetas no dan problema. Las *imágenes*, en cambio, no son de COCO para licenciarlas: son fotos de Flickr bajo un mosaico de condiciones individuales, que el consorcio COCO explícitamente no posee. La mayor parte del sector lo trata como un riesgo bajo y sigue adelante, pero "COCO no da problemas" es una afirmación solo sobre las anotaciones. Más allá de COCO, la cosa se endurece rápido: muchos datasets aéreos, médicos y de conducción son no comerciales, y una licencia de código permisiva en un modelo no los blanquea.

Audita las tres capas, siempre.

## Una nota sobre MIT frente a Apache-2.0

Distribuimos una biblioteca con licencia MIT, así que nos vendría bien decirte que MIT es sin más la mejor licencia. La comparación honesta es más interesante.

MIT no está libre de obligaciones: debes conservar el aviso de copyright y el texto de la licencia en las copias que distribuyas. Es una condición real, solo que barata. Y Apache-2.0 tiene algo que MIT no tiene: una **concesión expresa de patentes** de cada contribuidor, más una cláusula de represalia que extingue la concesión de quien demande en relación con la obra. MIT no dice nada sobre patentes. Para una empresa cuyo principal temor es la exposición a patentes, Apache-2.0 es posiblemente la *más segura* de las dos, y buena parte del ecosistema permisivo de detección (YOLOX, RT-DETR, D-FINE, DEIM) es Apache-2.0 exactamente por eso.

Lo que MIT te aporta es simplicidad y compatibilidad. Ninguna de las dos licencias te obligará nunca a publicar tu código fuente, que es el eje del que trata realmente este artículo. Si alguien te dice que MIT frente a Apache es la decisión importante, te está vendiendo algo. La decisión importante es permisiva frente a copyleft.

## Entonces, ¿qué YOLO puedes usar realmente?

- **Producto comercial de código cerrado:** YOLOv7 o YOLOv9 a través del [repositorio MIT](https://github.com/MultimediaTechLab/YOLO) del laboratorio, YOLOX o PP-YOLOE (desde PaddleDetection) entre los originales, o un framework MIT mantenido como LibreYOLO si prefieres no montar y auditar todo esto por tu cuenta. Evita cualquier cosa GPL o AGPL, salvo que vayas a abrir toda tu aplicación o a comprar la licencia Enterprise.
- **Proyecto de código abierto:** cualquiera, siempre que tu licencia sea compatible. Si tu proyecto es AGPL, la línea de Ultralytics encaja de forma natural, y la investigación más reciente (YOLOv13, YOLO26, YOLOE) llega allí primero.
- **Investigación y benchmarking:** las licencias apenas te limitan. Usa lo que usara el paper con el que te comparas.
- **El plan de "ya lo resolveremos más adelante":** no funciona. Deshacer una dependencia AGPL cuando tu producto ya está construido implica volver a entrenar, volver a validar y volver a exportar todo, incluidos los pesos. Elige primero el repositorio, y por tanto la licencia.

Para una comparación de los propios frameworks en lugar de sus licencias, consulta [Las mejores alternativas a Ultralytics en 2026](/articles/best-ultralytics-alternatives).

## La opción MIT: lo que incluye LibreYOLO

Aviso: LibreYOLO es nuestro proyecto, y su licencia es la razón de que exista este artículo.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) es una única base de código con licencia MIT** que cubre los detectores de abajo tras una sola API, con entrenamiento y exportación a ONNX, TensorRT, OpenVINO y NCNN integrados. Sin copyleft, sin cláusula de red, sin nivel enterprise.

Esta es la gama completa de detección de objetos, con el upstream de cada familia y su licencia, para que veas exactamente de dónde sale cada vía permisiva:

| En LibreYOLO | Modelo | Upstream | Licencia upstream |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Los YOLO de la era Darknet, en PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Dominio público |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (la reescritura del propio laboratorio, no el repositorio GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, más una variante end-to-end sin NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (la reescritura del propio laboratorio, no el repositorio GPL) |
| `LibreYOLO9P2` | YOLOv9 con una cabeza de stride 4 para objetos pequeños | Original de LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, detección y pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Código Apache-2.0. Los pesos upstream restringidos **no** se redistribuyen |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR y v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, detección, segmentación, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 para N/S/M/L. XL/2XL están bajo la Platform Model License de Roboflow, así que solo distribuimos los tamaños Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Código Apache-2.0. Los tamaños más grandes usan el backbone DINOv3 de Meta bajo la licencia propia de Meta, no aprobada por la OSI, que permite la redistribución pero **prohíbe el uso militar, nuclear, de espionaje y de armamento**. Esos tamaños se publican como `other`, no como Apache. Léela antes de distribuir cualquier cosa de doble uso |
| `LibreRTMDet` | RTMDet ([sin MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (el hermano Apache, no el mmyolo GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Arquitectura de [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoints a través del re-port [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (ambos) |
| `LibreEC` | EdgeCrafter, detección, pose, segmentación | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Detector de centroides para hardware de clase microcontrolador | Original de LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, vocabulario abierto (inferencia) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, vocabulario abierto (inferencia) | Google, a través de `transformers` | Apache-2.0 |

Hay dos cosas sobre las que esa columna es deliberadamente honesta.

**No se copia código fuente GPL ni AGPL en el código base.** Cada familia de arriba se construye a partir de un upstream de dominio público, MIT o Apache-2.0, que es la regla que mantiene el framework como MIT. Cuando una arquitectura tiene a la vez un hogar permisivo y uno copyleft, construimos a partir del permisivo: nuestros YOLOv9 y YOLOv7 proceden del repositorio MIT del laboratorio y no de los originales GPL, y nuestro RTMDet de MMDetection con licencia Apache y no de MMYOLO con licencia GPL. YOLO-World no está en la lista, aunque nos lo piden a menudo, porque es GPL-3.0 y no hay ningún sitio permisivo del que tomarlo.

**Permisivo no es lo mismo que sin restricciones, y preferimos decirlo antes de que lo descubras en una auditoría.** Los tamaños más grandes de DEIMv2 heredan las condiciones de DINOv3 de Meta, incluida la prohibición del uso militar, nuclear, de espionaje y de armamento, lo que es una restricción real si trabajas en algo cercano al doble uso. Un puñado de checkpoints de vista previa para investigación están entrenados con datasets no comerciales (DOTA, VisDrone) y están etiquetados como `cc-by-nc` en Hugging Face en lugar de distribuirse discretamente como si fueran de uso libre. Y la misma cautela que aplicamos a los pesos de los demás se aplica a los nuestros: nuestros checkpoints YOLO9 están convertidos a partir de los propios checkpoints del repositorio MIT del laboratorio, así que heredan lo que sea cierto de ellos, que, como se ha dicho arriba, es "afirmado, no auditado formalmente". La licencia MIT cubre el código que escribimos. Cada checkpoint lleva las condiciones de aquello con lo que y a partir de lo que se entrenó, publicadas para cada peso.

Más allá de la detección, la misma API cubre segmentación de instancias y semántica, pose, bounding boxes orientados, clasificación (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), profundidad monocular, restauración de imágenes y estimación de la mirada.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

El mismo flujo de trabajo de YOLO, sin los deberes de licencias.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Marcas comerciales y afiliación: YOLO, YOLOv8, YOLO11, YOLO26 y Ultralytics son marcas comerciales de sus respectivos titulares, incluida Ultralytics Inc. LibreYOLO es un proyecto independiente y no está afiliado, patrocinado ni respaldado por Ultralytics ni por ninguna otra organización mencionada en este artículo. Los nombres de productos y repositorios se usan únicamente para identificar y comparar el software del que se habla.*

*Este artículo es información general, no asesoramiento legal, vigente a 11 de julio de 2026, y escrito por uno de los proveedores de la comparación. Verifica la licencia actual antes de distribuir.*
