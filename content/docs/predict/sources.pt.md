---
title: Fontes de predição
seo_title: Fontes de predição no LibreYOLO
description: >-
  Todas as fontes que predict aceita: imagens, pastas, URLs, arquivos de vídeo,
  webcams, RTSP, YouTube, captura de tela, listas de imagens e arquivos
  .streams.
lead: >-
  O argumento source é classificado antes de qualquer coisa ser aberta, então
  uma única chamada dá conta de um JPEG, uma pasta, um MP4, um índice de webcam,
  uma URL RTSP, uma região da tela ou uma lista de câmeras.
keywords:
  - inferência de vídeo yolo python
  - rtsp
  - detecção de objetos webcam python
  - predict em uma pasta de imagens
  - detecção de objetos captura de tela
  - vários streams rtsp
  - arquivo streams
  - inferência youtube
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Classificação de fontes lida de libreyolo/utils/source.py (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Tipos de imagem aceitos e
  extensões de diretório de libreyolo/utils/image_loader.py. Extensões de vídeo
  e caminhos de gravação de libreyolo/utils/video.py. Sintaxe de screen de
  libreyolo/utils/screen.py. Formatos de retorno e valores padrão dos argumentos
  de InferenceRunner.__call__ em libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Uma imagem
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Uma fonte de imagem única retorna um Results, não uma lista.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Imagens em memória
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Uma pasta
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("sample_folder")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Uma pasta retorna uma lista, um Results por imagem, ordenada por
        caminho.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Um arquivo de vídeo (use um clipe seu)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Substitua clip.mp4 por um arquivo de vídeo no disco.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Um quadro a cada três, gravado em disco'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (precisa de uma câmera conectada)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Índice de webcam 0. Fontes ao vivo nunca terminam, então limite o
        loop.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (precisa de uma URL de câmera acessível)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Um arquivo .streams (use suas próprias câmeras)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Uma lista de câmeras
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Uma captura de tela (precisa de mss e de uma sessão de desktop)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sem stream=True isso pega um único quadro.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Uma região de um monitor, continuamente'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Como uma fonte é classificada

`classify_source` inspeciona o valor antes de qualquer coisa ser aberta ou
baixada, nesta ordem. A primeira regra que casar vence.

| Fonte | Lida como |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Captura de tela |
| Um `int` não negativo, ou uma string de dígitos sem nenhum arquivo com esse nome | Webcam |
| Uma URL `rtsp://`, `rtmp://`, `tcp://` ou `udp://` | Stream de rede |
| Uma URL `http(s)://` cujo caminho termina em `.m3u8` | Stream de rede |
| Uma URL de página do YouTube | Stream de rede |
| Uma lista ou tupla cujos itens são todos ao vivo ou vídeo | Vários streams ao vivo |
| Qualquer outra lista ou tupla | Batch de imagens |
| Um caminho terminado em `.streams` | Vários streams ao vivo |
| Um caminho com extensão de vídeo | Arquivo de vídeo |
| Um diretório existente | Pasta de imagens |
| Qualquer outra coisa | Imagem única |

Uma lista que mistura fontes ao vivo com imagens levanta `TypeError`. Um índice
de webcam negativo levanta `ValueError`.

O classificador nunca toca a rede, então uma URL digitada errada só aparece
quando a captura é aberta, não quando `predict` é chamado.

## Imagens

<code-tabs name="images" />

Uma fonte de imagem única aceita sete tipos.

| Tipo | Lido como |
|---|---|
| `str` ou `pathlib.Path` | Arquivo local, `http(s)://`, `s3://` ou `gs://` |
| `PIL.Image.Image` | Convertido para RGB |
| `numpy.ndarray` | Escala de cinza 2D, ou HWC ou CHW 3D; um array 4D usa sua primeira imagem |
| `torch.Tensor` | CHW ou NCHW, lido como RGB; um tensor em batch usa sua primeira imagem |
| `bytes` | Dados de imagem codificados |
| `io.BytesIO` | Dados de imagem codificados |

Tudo é convertido para RGB antes do pré-processamento. Os arrays NumPy são o
único caso em que a ordem dos canais é ambígua, então `color_format` controla
isso: `"auto"` (o padrão) deixa o array como está, `"bgr"` inverte os canais,
que é o que um quadro lido com OpenCV precisa.

Arrays de ponto flutuante são reescalados pela própria faixa: valores iguais ou
abaixo de `1.0` são multiplicados por 255, valores maiores são recortados para
`[0, 255]`. Um array RGBA descarta seu canal alfa.

Caminhos remotos precisam de um pacote cada, e nenhum deles vem instalado por
padrão: `requests` para `http(s)://`, `boto3` para `s3://` e `gcsfs` para
`gs://`.

## Pastas

Um diretório é varrido recursivamente e ordenado, e todo arquivo com um destes
sufixos vira uma imagem: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`,
`.tiff`, `.tif`. Qualquer outra coisa na pasta é ignorada. Uma pasta vazia
retorna uma lista vazia em vez de levantar erro.

Pastas e listas são as duas fontes que aceitam `batch`, que roda um forward pass
empilhado por bloco nas famílias que suportam isso. Veja
[Desempenho da inferência](/docs/predict/performance).

## Arquivos de vídeo

<code-tabs name="video" />

Um caminho conta como vídeo quando seu sufixo é um de `.asf`, `.avi`, `.gif`,
`.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` aparece nas duas listas. Um caminho `.gif` passado direto para `predict` é
aberto como vídeo, porque a checagem de vídeo roda primeiro; um `.gif` que está
dentro de uma pasta varrida é carregado como imagem estática.

`vid_stride` processa um quadro a cada N e vale `1` por padrão. Sem
`stream=True` o vídeo inteiro é decodificado em uma lista, e qualquer coisa
acima de 500 quadros depois do stride emite um aviso sugerindo `stream=True`.

Cada `Results` vindo de um vídeo carrega `frame_idx`.

## Webcams, streams de rede e YouTube

<code-tabs name="live" />

Fontes ao vivo são ilimitadas, então exigem `stream=True`. Sem ele, `predict`
levanta `ValueError` em vez de tentar juntar uma lista infinita.

Os quadros são lidos em uma thread de segundo plano, uma por captura. Por padrão
a fila guarda apenas o quadro mais recente, então um modelo mais lento que a
câmera pula quadros em vez de ficar para trás. `stream_buffer=True` mantém todo
quadro capturado, o que os preserva ao custo de uma latência crescente.

Um índice de webcam é um `int` ou uma string de dígitos. No Windows a captura é
aberta primeiro pelo backend DirectShow e cai no backend padrão se isso falhar.

URLs de página do YouTube são resolvidas para uma URL de mídia direta sem baixar
o vídeo, o que precisa do `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

As labels de stream são mascaradas antes de irem para o log ou de serem usadas
como nome de arquivo. Uma URL que carrega credenciais aparece como
`user:***@host`, e as query strings são descartadas das labels de stream direto
porque URLs assinadas e bearer tokens ficam ali. O id de vídeo do YouTube é
mantido, já que não é uma credencial.

## Várias câmeras de uma vez

<code-tabs name="streams" />

Um arquivo `.streams` é uma fonte por linha. Linhas em branco e linhas que
começam com `#` são ignoradas. Cada linha restante precisa ser, ela mesma, um
índice de webcam, um stream de rede, uma URL do YouTube ou o caminho de um
arquivo de vídeo; qualquer outra coisa levanta `ValueError` informando o número
da linha. Um arquivo vazio levanta erro em vez de começar sem nenhuma câmera.

Uma lista ou tupla de fontes ao vivo faz a mesma coisa sem arquivo nenhum.

Cada captura ganha sua própria thread, e os quadros de todas elas são
multiplexados em um único gerador. Cada passada consulta cada stream ativo e
entrega o que estiver pronto, então uma câmera lenta não segura uma rápida, e os
quadros de câmeras diferentes se intercalam. Um stream que termina sai do rodízio
enquanto os outros continuam.

## Captura de tela

<code-tabs name="screen" />

Uma fonte de tela é a palavra `screen` seguida de zero, um, quatro ou cinco
inteiros. Qualquer outra quantidade levanta `ValueError`.

| Forma | Captura |
|---|---|
| `"screen"` | Todos os monitores, mesclados |
| `"screen 1"` | O monitor 1 |
| `"screen 100 200 512 256"` | Uma caixa no desktop mesclado |
| `"screen 1 100 200 512 256"` | Uma caixa no monitor 1 |

As coordenadas da caixa são `left top width height`, relativas ao canto superior
esquerdo do monitor escolhido. Uma fonte de tela informa sua taxa de quadros como
30 dividido por `vid_stride`, que é a taxa com que um vídeo salvo é gravado. A
captura precisa do pacote `mss`:

```bash
pip install mss
```

Sem `stream=True`, uma fonte de tela pega um quadro e retorna um único
`Results`, que é o equivalente, em captura de tela, a predizer sobre um arquivo
de imagem. Com `stream=True` ela captura até o loop ser interrompido.

## O que predict retorna

O formato do valor de retorno depende da fonte e de `stream`.

| Fonte | `stream=False` | `stream=True` |
|---|---|---|
| Imagem única | Um `Results` | Gerador de um `Results` |
| Lista de imagens | Lista de `Results` | Gerador |
| Pasta | Lista de `Results` | Gerador |
| Arquivo de vídeo | Lista de `Results` | Gerador |
| Tela | Um `Results` | Gerador, ilimitado |
| Webcam, stream de rede, `.streams` | `ValueError` | Gerador, ilimitado |

Uma imagem única retorna o próprio objeto `Results`. Indexá-lo seleciona uma
detecção, não uma imagem, então `result[0]` em uma predição de imagem única é o
primeiro box, não a primeira figura. Para saber o que esses objetos carregam,
veja [Trabalhando com resultados](/docs/predict/results).

## Onde save grava

`save=True` grava a saída anotada em um diretório de execução em vez de
retorná-la.

As imagens vão para um `runs/detect/predict`, `runs/detect/predict2` e assim por
diante, com incremento automático, mantendo o nome do arquivo de origem. Toda
imagem de um mesmo processo cai no mesmo diretório, então duas pastas de entrada
com um mesmo nome de arquivo sobrescrevem uma à outra. Imagens em memória não têm
nome de arquivo para reaproveitar e são numeradas `image0`, `image1` e assim por
diante.

Fontes de vídeo e ao vivo são gravadas como um único `.mp4` nomeado a partir da
fonte.

`output_path` sobrescreve o diretório. Um caminho com sufixo é tratado como
arquivo, um caminho sem sufixo como diretório. `output_file_format` seleciona a
codificação da imagem estática e aceita `jpg`, `png` ou `webp`.

Depois de salvar, o caminho gravado também é anexado ao resultado como
`result.saved_path`.
