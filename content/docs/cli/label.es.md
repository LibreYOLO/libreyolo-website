---
title: libreyolo label
seo_title: "Referencia del comando libreyolo label"
description: "Arranca la herramienta local de anotación de bounding boxes: argumentos con sus valores por defecto, el interruptor de asistencia por IA y qué expone enlazar con una interfaz de red."
lead: "Arranca una herramienta web local para dibujar y editar bounding boxes. Escribe ficheros de etiquetas en el formato nativo de LibreYOLO, así que un dataset anotado aquí se entrena sin ningún paso de conversión."
keywords: [libreyolo label cli, herramienta de etiquetado bounding box, etiquetar dataset yolo, auto etiquetado cli, compartir libreyolo label]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo label
    mono: true
  - label: Salida
    value: "Una URL de servidor en stdout; las etiquetas se escriben como labels/*.txt junto a las imágenes"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        # Abre la página de inicio del proyecto; elige o crea un dataset en el navegador.
        libreyolo label
    - label: Solo manual, puerto fijo
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Dejar que se unan tus compañeros
      language: bash
      code: |
        libreyolo label share=true
---

## Sinopsis

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, así que
`port=9200` y `--port 9200` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `data` | | YAML o carpeta del dataset que se abre directamente. Arranca en la página de inicio del proyecto si no se indica |
| `host` | `127.0.0.1` | Host o interfaz al que enlazar |
| `port` | `8000` | Puerto al que enlazar. Salta al siguiente libre si está ocupado |
| `device` | `auto` | Dispositivo para el auto-etiquetado por IA: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Desactiva el auto-etiquetado por IA y deja un etiquetador manual |
| `no_browser` | `false` | No abrir el navegador automáticamente |
| `share` | `false` | Enlaza en `0.0.0.0` para que tus compañeros de red puedan unirse |
| `json` | `false` | Salida JSON en stdout |
| `quiet` | `false` | Silencia stderr |
| `verbose` | `false` | Salida detallada en stderr |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Qué escribe

Los boxes se guardan como ficheros `labels/*.txt` en el formato nativo de
LibreYOLO, que es el que lee `libreyolo train`, así que después no hay que
convertir nada. Esta versión solo maneja bounding boxes. Las ediciones se
guardan a medida que te mueves entre imágenes.

### Abrir un dataset

Sin `data`, la herramienta arranca en la página de inicio del proyecto y el
dataset se elige o se crea desde el navegador. Pasar `data=path/to/data.yaml`
abre ese dataset directamente, y la línea de arranque informa del número de
imágenes, del número de clases y de si el dataset es escribible. Un dataset de
solo lectura también se abre e indica por qué no se puede escribir en él.

### Compartir, y qué hace `host`

`share=true` enlaza la dirección comodín, lo que permite que otras máquinas de
tu red lleguen a la herramienta mientras que las acciones administrativas
—cambiar o borrar proyectos y lanzar cómputo— se quedan en esta máquina.

Poner `host` en una interfaz concreta hace algo distinto y menos seguro: el host
pasa a ser indistinguible de un cliente de red, así que todos los clientes
obtienen derechos administrativos. El comando imprime un aviso en stderr cuando
lo haces. Es preferible `share=true`.

### Puertos y apagado

Un puerto ocupado pasa al siguiente, hasta veinte por encima del solicitado. Si
fallan los veinte, se sale con `io_error`. La URL impresa en stdout corresponde
al puerto que se enlazó realmente. Con `share=true`, el resultado incluye además
`lan_url`, la dirección que deben abrir tus compañeros.

El comando se sirve en primer plano hasta que pulsas Ctrl+C.

Relacionado: [`libreyolo doctor`](/docs/cli/doctor) para comprobar el dataset
etiquetado antes de entrenar, y [`libreyolo train`](/docs/cli/train) para
entrenar con él.
