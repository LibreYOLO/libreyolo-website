---
title: Triton Inference Server
seo_title: Servir um modelo LibreYOLO no NVIDIA Triton
description: >-
  Sirva uma exportação ONNX do LibreYOLO através do NVIDIA Triton: o layout do
  repositório de modelos, o config.pbtxt gerado e a predição contra uma URL de
  modelo HTTP.
lead: >-
  O Triton Inference Server hospeda um repositório de modelos e responde a
  requisições de inferência por HTTP. O LibreYOLO exporta o grafo ONNX, gera um
  config.pbtxt que carrega os metadados da exportação como um único parâmetro do
  Triton e trata uma URL de modelo como um caminho de modelo carregável.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - repositório de modelos triton
  - inferência yolo remota
last_verified: 1.5.0
meta:
  - label: Chamada
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Auxiliar
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protocolo
    value: >-
      Somente inferência V2 por HTTP e HTTPS. Sem gRPC, autenticação, memória
      compartilhada ou carga e descarga de modelos.
  - label: Timeouts
    value: Os timeouts de conexão e de rede são de 30 segundos por padrão
verification: >-
  Lido de libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md e pyproject.toml no branch dev. Os comandos de container são os
  fixados em docs/triton.md.
snippets:
  install:
    - label: Instalação
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Exportar no layout do repositório
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Gerar o config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Layout resultante
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Iniciar o servidor
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Esperar ficar pronto
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Parar o servidor
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Fazer predição no modelo servido
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Comparar com o modelo local
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 'Fixar uma versão, ou mudar o timeout'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Um segundo segmento de caminho seleciona a versão do modelo. Sem ele,
        # quem escolhe é a política de versão configurada no Triton.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Os timeouts de conexão e de rede são de 30 segundos por padrão.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Instalação

<code-tabs name="install" />

O extra `triton` instala o `tritonclient[http]`. Os extras de gRPC e de memória
compartilhada ficam de fora de propósito: esta integração é somente inferência V2
por HTTP e HTTPS. O `onnx` é necessário porque tanto o artefato servido quanto o
gerador de config trabalham a partir de um grafo ONNX.

## Construir o repositório de modelos

Exporte com um eixo de batch dinâmico, no layout de diretórios que o Triton espera.

<code-tabs name="repo" />

O Triton não preserva os metadados customizados do ONNX na resposta de model-config,
então os metadados exportados completos precisam viajar de outra forma. O
`create_triton_config` codifica tudo como um único parâmetro string JSON chamado
`libreyolo_metadata` no `config.pbtxt`, emite as declarações de entrada e saída na
ordem do grafo, cuida do escape do JSON e fixa o modelo em `KIND_CPU`.

O auxiliar valida antes de escrever. Ele exige exatamente uma entrada no grafo ONNX,
pelo menos uma saída, formas de tensor resolvíveis e metadados cujo mapa `names`
defina todo índice de classe de 0 até `nc - 1`. Um modelo que falhe em qualquer uma
dessas checagens é rejeitado na hora do config, e não na primeira requisição.

`max_batch_size: 8` combina com uma exportação dinâmica e deixa o servidor agrupar
até oito imagens por requisição. Para um grafo ONNX de batch 1 fixo use
`max_batch_size=0`; o LibreYOLO então envia as imagens sequencialmente.

## Iniciar o servidor

<code-tabs name="serve" />

Os comandos fixam o Triton Server 26.04 e omitem deliberadamente as flags de GPU do
Docker, já que o `KIND_CPU` no config gerado impede a alocação em GPU de qualquer
jeito.

## Executar o artefato

Uma URL de modelo do Triton é um caminho de modelo. `LibreYOLO()` verifica se há um
esquema `http` ou `https` antes de qualquer tratamento de caminho local e retorna um
backend que conversa com o servidor, então o ponto de chamada é idêntico ao de um
checkpoint local, e o objeto `Results` que volta também.

<code-tabs name="run" />

A forma da URL é `http(s)://host:porta/modelo` com um segmento de versão opcional. A
porta precisa ser explícita. Credenciais embutidas, query string e fragmento são
todos rejeitados, assim como um caminho com mais de dois segmentos.

`device` é aceito e ignorado com uma linha de log, porque a alocação é decisão do
servidor.

## Restrições

O backend falha com um erro direto em vez de entregar um resultado degradado quando
o contrato não é cumprido: metadados do LibreYOLO ausentes no config do modelo, mais
de uma entrada de modelo, divergência entre as saídas configuradas e os metadados do
modelo, um datatype de entrada que ele não suporta, ou um servidor ou modelo que não
está pronto.

Fora do contrato nesta versão: gRPC, autenticação, memória compartilhada e carregar
ou descarregar modelos pela API.

Qualquer formato que o próprio Triton suporte pode ser servido, mas o parâmetro de
metadados e o config gerado têm formato de ONNX aqui, então o caminho do LibreYOLO é
[ONNX](/docs/export/onnx) para dentro do repositório. Para um pipeline de vídeo
completo, em vez de um servidor de requisição-resposta, veja o
[DeepStream](/docs/export/deepstream).
