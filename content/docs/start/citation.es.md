---
title: Cita
seo_title: "Cómo citar LibreYOLO y a los autores originales"
description: "Cómo citar LibreYOLO en un artículo y cómo citar a los autores de la familia de modelos que ejecutaste. Ambas cosas van en la misma sección de métodos."
lead: "Una cita completa de LibreYOLO tiene dos partes: la biblioteca y el trabajo publicado que hay detrás de la familia de modelos que produjo el resultado."
keywords: [citar libreyolo, libreyolo bibtex, libreyolo citation cff, como citar un modelo de vision, citar paper de deteccion de objetos, referencia bibliografica vision por computador]
last_verified: "1.5.0"
---

## Cómo citar LibreYOLO

El repositorio publica sus metadatos de cita como
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
no como un bloque BibTeX. GitHub lee ese archivo y ofrece un botón Cite this
repository en la página del repositorio, que genera APA y BibTeX a partir de él.
Toma la entrada de ahí en lugar de escribir una a mano.

El archivo completo:

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

No lleva versión ni fecha de publicación a propósito.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
indica a los maintainers que nunca cambien la versión, la fecha ni el título de
`CITATION.cff` ni de `.zenodo.json` durante una release, para que todas las citas
apunten a un único registro en lugar de dispersarse entre versiones. Indica en tu
propio texto la versión que ejecutaste y deja la cita como está.

## Cómo citar la familia de modelos

LibreYOLO es un port. Ejecutar `LibreRFDETRm.pt` significa ejecutar RF-DETR, y las
personas que escribieron RF-DETR son a quienes un revisor espera ver acreditadas.
Citar solo la biblioteca atribuye su trabajo al proyecto equivocado.

Todo lo necesario está en la página de la familia. La fila Upstream de la cabecera
nombra el trabajo original y la organización que hay detrás, y enlaza el paper y el
repositorio de código fuente. La sección Cita, más abajo, contiene el BibTeX.

Ese BibTeX está copiado literalmente del bloque de cita de los propios autores,
normalmente la sección Citation del README upstream o un `CITATION.cff`, y se
muestra con un enlace al bloque del que procede para que puedas contrastarlo con la
fuente. Nunca se ensambla a partir de los metadatos del paper. Una entrada
reconstruida a mano falla de forma silenciosa y cara: un coautor perdido, el
congreso equivocado, el tipo de entrada equivocado, un año que pertenece al
preprint. Los preprints también acaban siendo aceptados, así que una entrada puede
ser un `@inproceedings` aunque la versión que leíste estuviera en arXiv.

Copia el bloque tal cual. Si tu estilo bibliográfico necesita otro tipo de entrada,
convierte la entrada en lugar de volver a escribirla, y mantén la lista de autores
en su orden original.

## Qué necesita una sección de métodos

Tres cosas hacen que un resultado de LibreYOLO sea reproducible y esté
correctamente atribuido:

- La biblioteca, citada desde `CITATION.cff`, junto con la versión que ejecutaste.
  `libreyolo version` la imprime, junto con las versiones de Python, torch y CUDA
  contra las que se está ejecutando.
- El trabajo upstream, citado desde la sección Cita de la página de la familia.
- El nombre exacto del archivo del checkpoint, como `LibreRFDETRm.pt`. Los tamaños
  dentro de una familia se comportan de forma distinta, y varias familias publican
  checkpoints entrenados con datasets distintos bajo el mismo prefijo, así que el
  nombre de la familia por sí solo no identifica qué se ejecutó.

La atribución es además una condición de licencia para buena parte de lo que
publica LibreYOLO. Tanto Apache-2.0 como la familia CC BY exigen que el aviso viaje
con los pesos que redistribuyes, lo que es una obligación distinta de citar un
paper. Consulta [licencias](/docs/licensing) para ver qué términos aplican a cada
checkpoint.
