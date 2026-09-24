---
title: "¿YOLO es gratis para uso comercial? Licencias de YOLOv8, YOLO11 y YOLO26"
description: "YOLOv5, YOLOv8, YOLO11 y YOLO26 se distribuyen bajo AGPL-3.0, así que no son gratis para uso comercial de código cerrado. Aquí explicamos qué exige realmente la licencia, versión por versión, y cuál es la alternativa con licencia MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "¿YOLOv8 es gratis para uso comercial?"
    a: "No para productos de código cerrado. YOLOv8 es AGPL-3.0, así que debes publicar el código fuente de toda tu aplicación bajo AGPL-3.0 o comprar la licencia Enterprise de pago del proveedor. Para un uso comercial gratuito y sin restricciones, usa una biblioteca con licencia MIT como LibreYOLO."
  - q: "¿Bajo qué licencia está YOLO?"
    a: "Las versiones principales, YOLOv5, YOLOv8, YOLO11 y YOLO26, son AGPL-3.0 por defecto. AGPL-3.0 es una licencia copyleft fuerte: es de código abierto, pero no es gratuita para usarla dentro de un producto propietario."
  - q: "¿Puedo usar YOLO en un producto comercial de código cerrado?"
    a: "No bajo AGPL-3.0 sin cumplirla, y cumplirla significa publicar todo tu producto como código abierto bajo AGPL-3.0. Para mantener tu código privado, o compras la licencia Enterprise del proveedor o cambias a un framework con licencia permisiva como LibreYOLO (MIT)."
  - q: "¿Existe un YOLO sin AGPL?"
    a: "Sí. LibreYOLO tiene licencia MIT, así que no hay copyleft de AGPL ni cláusula de uso en red. Puedes usarlo libremente en software comercial y de código cerrado sin pagar nada."
  - q: "¿Cuánto cuesta una licencia comercial de YOLO?"
    a: "La licencia Enterprise del proveedor es un acuerdo de pago por empresa, con precios no públicos. LibreYOLO no cuesta nada: la licencia MIT concede el uso comercial gratis."
  - q: "¿Ejecutar YOLO solo en mis propios servidores evita la AGPL?"
    a: "No. La sección 13 de AGPL-3.0 cubre la interacción a través de una red. Si los usuarios acceden a tu versión modificada a través de una red, debes ofrecerles el código fuente correspondiente. Ejecutarlo solo en tus propios servidores no es una exención."
---

**No. Los modelos YOLO populares, YOLOv5, YOLOv8, YOLO11 y el nuevo YOLO26, no son gratis para uso comercial de código cerrado.** Se distribuyen bajo **AGPL-3.0**, una licencia copyleft fuerte. Si construyes un producto sobre uno de ellos, debes publicar *toda* tu aplicación como código abierto bajo AGPL-3.0, o comprar la licencia Enterprise de pago del proveedor. Si ninguna de las dos opciones te sirve, existe una alternativa con licencia permisiva: **[LibreYOLO](/) tiene licencia MIT y es gratis para cualquier uso comercial, sin ataduras.**

Si has buscado "¿YOLOv8 es gratis para uso comercial?", "licencia YOLO" o "YOLO sin AGPL", esta página lo responde con precisión, versión por versión.

## Licencias de YOLO de un vistazo

| Modelo | Licencia por defecto | ¿Gratis para uso comercial de **código cerrado**? | Qué tienes que hacer |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | No | Publicar toda tu app como código abierto, o comprar una licencia Enterprise |
| YOLOv8 | AGPL-3.0 | No | Publicar toda tu app como código abierto, o comprar una licencia Enterprise |
| YOLO11 | AGPL-3.0 | No | Publicar toda tu app como código abierto, o comprar una licencia Enterprise |
| YOLO26 | AGPL-3.0 | No | Publicar toda tu app como código abierto, o comprar una licencia Enterprise |
| **LibreYOLO** | **MIT** | **Sí** | **Nada. Inclúyelo en productos propietarios, gratis.** |

La versión en una línea: los modelos YOLO principales son de código abierto, pero bajo **AGPL-3.0**, una licencia copyleft que la mayoría de las empresas no pueden cumplir en un producto propietario. LibreYOLO te ofrece el mismo flujo de trabajo de YOLO bajo la **licencia MIT**, que no te impone ninguna obligación de ese tipo.

## ¿Bajo qué licencia está YOLO?

Los modelos YOLO más usados, **YOLOv5, YOLOv8, YOLO11 y el más reciente, YOLO26**, los distribuye su proveedor bajo la **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 es *copyleft fuerte*. No es el código abierto permisivo del tipo "haz lo que quieras", como MIT o Apache-2.0. Conlleva una obligación concreta y de gran alcance, y esa obligación es precisamente la que hace tropezar a los usuarios comerciales.

## ¿YOLOv8 es gratis para uso comercial?

No en el sentido en que lo entiende la mayoría. *Puedes* descargar YOLOv8 y usarlo comercialmente, pero solo si cumples la AGPL-3.0. En la práctica, eso significa:

- Si distribuyes, entregas o alojas un producto que incluye YOLOv8, debes **publicar el código fuente completo de ese producto**, tu app, tu código de integración, tus scripts y tu configuración, bajo AGPL-3.0.
- Esa obligación no se activa solo al entregar software a los usuarios, sino también al permitir que los usuarios interactúen con él **a través de una red**, la cláusula de la sección 13 que define a la AGPL. Un SaaS o un backend de API no se libran.

Para un proyecto personal o una investigación académica, eso no suele ser un problema. Para un producto comercial propietario, publicar todo tu código es casi siempre inviable. Todo lo anterior se aplica igual a **YOLO11** y **YOLO26**: misma licencia, misma obligación.

## ¿Y YOLOv5, YOLO11 y YOLO26?

La misma historia. Todos usan **AGPL-3.0** por defecto. La licencia Enterprise del proveedor abarca explícitamente "YOLO26, earlier YOLO versions, and any future YOLO models" (YOLO26, versiones anteriores de YOLO y cualquier modelo YOLO futuro), lo que indica que la política de licencias es deliberada y coherente en toda la familia. Elegir una versión más nueva no te da una licencia más favorable.

Para ser exhaustivos: no *todos* los modelos con "YOLO" en el nombre son AGPL. Algunas variantes de la comunidad son GPL-3.0 o Apache-2.0, y las licencias varían según el autor. Pero las versiones a las que se refiere la gente cuando escribe "YOLOv8" o "YOLO11" son AGPL-3.0.

## Qué exige realmente AGPL-3.0, en lenguaje claro

AGPL-3.0 amplía la GPL con un añadido crucial: la **cláusula de uso en red**. La lista práctica:

1. **Distribuye el código fuente.** Si transmites software que incluye código AGPL, debes poner a disposición el *código fuente correspondiente completo* de toda la obra combinada, bajo AGPL-3.0.
2. **El uso en red cuenta.** Si los usuarios interactúan con tu versión modificada a través de una red, ya sea una aplicación web, una API o un SaaS, debes ofrecerles ese mismo código fuente. No existe el resquicio de "solo lo ejecutamos en nuestros servidores".
3. **El copyleft es viral.** La obligación alcanza a *toda* la obra derivada, no solo a los archivos de YOLO. Tu lógica de negocio queda dentro del ámbito de la AGPL.

Esto es información general, no asesoramiento legal, así que consulta a un abogado para tu caso concreto. Pero la conclusión es sencilla: **AGPL-3.0 y el software comercial de código cerrado no son compatibles.**

## La opción de la licencia Enterprise, y su inconveniente

El proveedor ofrece una **licencia Enterprise** de pago que elimina las obligaciones de la AGPL y te permite integrar YOLO en un producto propietario sin publicar nada como código abierto. Es una vía legítima y, para algunas empresas, la adecuada.

Los inconvenientes:

- **Cuesta dinero**: una tarifa comercial recurrente, negociada con cada empresa, con precios no públicos.
- **Es una dependencia de las condiciones de un único proveedor**, que pueden cambiar y que solo cubren el código de ese proveedor.
- **Estás pagando por un permiso** para usar una arquitectura de modelo que, en esencia, es investigación publicada.

Si una tarifa recurrente y la dependencia de un proveedor te resultan aceptables, la licencia Enterprise funciona. Si prefieres no pagar, o no cargar con la obligación en absoluto, sigue leyendo.

## La alternativa MIT: LibreYOLO

Aviso: LibreYOLO es nuestro. También es la razón por la que esta página puede terminar con una respuesta clara en lugar de un encogimiento de hombros.

**LibreYOLO es un framework de YOLO publicado bajo la licencia permisiva MIT.** Ese único hecho lo cambia todo en el uso comercial:

- **Gratis para uso comercial**, incluidos productos propietarios de código cerrado.
- **Sin copyleft ni cláusula de red.** Nunca tendrás que publicar tu app como código abierto.
- **Sin tarifas por puesto ni de tipo enterprise.** MIT significa MIT.
- **El mismo flujo de trabajo que ya conoces**, así que migrar es sencillo.

LibreYOLO es un framework real y mantenido, no un wrapper: detección de objetos, segmentación, pose, clasificación, profundidad y más, bajo una API familiar, con entrenamiento y exportación a ONNX/TensorRT/OpenVINO/NCNN integrados. La diferencia no está en el flujo de trabajo. La diferencia está en la licencia.

Si llegas aquí desde la pregunta más amplia de "abandonar el YOLO con AGPL", escribimos una comparación más completa del ecosistema en [Las mejores alternativas a Ultralytics en 2026](/articles/best-ultralytics-alternatives).

Para la licencia de todos los demás YOLO, desde los originales de Darknet en dominio público hasta YOLOv9, YOLOv10, YOLOv12, YOLOv13 y YOLO26, consulta [Todas las licencias de YOLO, explicadas](/articles/yolo-licenses-explained).

### Por qué MIT importa para una empresa

La licencia MIT te permite usar, modificar, integrar y vender software construido sobre LibreYOLO **sin obligación de revelar tu código fuente** y **sin pagar nada**. Es la licencia detrás de gran parte del stack de software moderno precisamente porque es segura para la adopción comercial. El producto es tuyo; no debes nada.

## Pruébalo

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

Conservas la experiencia de desarrollo de YOLO: entrenar con tus datos, predecir y exportar para desplegar. Te quitas de encima la obligación de la AGPL y la factura enterprise.

LibreYOLO tiene licencia MIT, funciona en Linux, Mac y Windows, y corre en GPU, Apple Silicon y CPU sin cambiar el código. Una sola API abarca YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, la línea RT-DETR, segmentación, pose, profundidad y más, todo con licencias permisivas.

Dale una estrella en GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentación: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
