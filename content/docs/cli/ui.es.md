---
title: libreyolo ui
seo_title: "referencia del comando libreyolo ui"
description: "Arranca la interfaz web local de inferencia: dirección de enlace, comportamiento del puerto, selección de dispositivo y cómo termina el comando."
lead: "Arranca un servidor web local que acepta imágenes soltadas o pegadas, ejecuta sobre ellas el modelo que elijas y muestra los resultados en el navegador."
keywords: [libreyolo ui cli, interfaz web libreyolo, inferencia local navegador, inferencia arrastrando imagenes, puerto libreyolo ui]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo ui
    mono: true
  - label: Salida
    value: "Una URL de servidor por stdout y, a continuación, el proceso se queda en primer plano"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo ui
    - label: Puerto fijo, sin navegador
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: En la CPU, salida legible por máquina
      language: bash
      code: |
        libreyolo ui device=cpu json=true
---

## Sinopsis

```bash
libreyolo ui [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, así que
`port=9000` y `--port 9000` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `host` | `127.0.0.1` | Host o interfaz al que enlazar |
| `port` | `8000` | Puerto al que enlazar. Salta al siguiente libre si está ocupado |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | No abrir el navegador automáticamente |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silenciar stderr |
| `verbose` | `false` | Salida detallada por stderr |

## Ejemplos

<code-tabs name="examples" />

## Notas

El enlace por defecto es loopback, así que la interfaz solo es accesible desde
esta máquina.

Si el puerto solicitado está ocupado, el comando prueba el siguiente y sigue
subiendo hasta veinte puertos por encima del pedido. Si fallan los veinte,
termina con `io_error` y la sugerencia de indicar otro puerto. La URL impresa
por stdout corresponde al puerto que se enlazó realmente, así que léela en vez
de dar por hecho el que pediste.

Salvo que uses `no_browser=true`, se abre una pestaña del navegador en esa URL
poco después del enlace.

El comando sirve entonces en primer plano hasta que pulses Ctrl+C, que apaga el
servidor de forma limpia. No hay modo desacoplado; mándalo a segundo plano con
tu shell si quieres recuperar la terminal.

`json=true` imprime la URL y el dispositivo como un único objeto con
`schema_version` antes de que arranque el servidor, que es como un script
averigua el puerto enlazado.

Relacionado: [`libreyolo label`](/docs/cli/label) para dibujar cajas y guardar
etiquetas, [`libreyolo monitor`](/docs/cli/monitor) para seguir entrenamientos.
Ambos son servidores web locales con el mismo comportamiento de puerto y
navegador.
