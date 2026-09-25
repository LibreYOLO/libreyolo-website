---
title: YOLO é grátis para uso comercial? Licenças do YOLOv8, YOLO11 e YOLO26
description: "YOLOv5, YOLOv8, YOLO11 e YOLO26 são distribuídos sob AGPL-3.0, então não são gratuitos para uso comercial em software de código fechado. Veja o que a licença exige, versão por versão, e conheça a alternativa com licença MIT."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "O YOLOv8 é grátis para uso comercial?"
    a: "Não para produtos de código fechado. O YOLOv8 usa AGPL-3.0, então você precisa publicar o código-fonte completo da sua aplicação sob AGPL-3.0 ou comprar a Enterprise License paga do fornecedor. Para uso comercial gratuito e sem restrições, use uma biblioteca com licença MIT, como o LibreYOLO."
  - q: "Qual é a licença do YOLO?"
    a: "As versões mais usadas, YOLOv5, YOLOv8, YOLO11 e YOLO26, são distribuídas sob AGPL-3.0 por padrão. AGPL-3.0 é uma licença copyleft forte: é código aberto, mas não permite o uso gratuito dentro de um produto proprietário."
  - q: "Posso usar YOLO em um produto comercial de código fechado?"
    a: "Não sob AGPL-3.0 sem cumprir as exigências, o que significa abrir o código-fonte de todo o produto sob AGPL-3.0. Para manter seu código privado, você precisa comprar a Enterprise License do fornecedor ou mudar para um framework com licença permissiva, como o LibreYOLO (MIT)."
  - q: "Existe um YOLO sem AGPL?"
    a: "Sim. O LibreYOLO usa a licença MIT, então não há copyleft da AGPL nem cláusula de uso em rede. Você pode usá-lo gratuitamente em software comercial e de código fechado, sem pagar."
  - q: "Quanto custa uma licença comercial do YOLO?"
    a: "A Enterprise License do fornecedor é um contrato pago por empresa, com preços não divulgados. O LibreYOLO não custa nada: a licença MIT permite o uso comercial gratuitamente."
  - q: "Executar YOLO apenas nos meus próprios servidores evita a AGPL?"
    a: "Não. A Seção 13 da AGPL-3.0 trata da interação pela rede. Se os usuários acessarem sua versão modificada pela rede, você precisa oferecer a eles o código-fonte correspondente. Executar o software apenas nos seus próprios servidores não é uma exceção."
---

**Não. Os modelos YOLO mais populares, YOLOv5, YOLOv8, YOLO11 e o novo YOLO26, não são gratuitos para uso comercial em software de código fechado.** Eles são distribuídos sob **AGPL-3.0**, uma licença copyleft forte. Se você criar um produto com um deles, precisa abrir o código-fonte de *toda* a aplicação sob AGPL-3.0 ou comprar a Enterprise License paga do fornecedor. Se nenhuma dessas opções funcionar para você, existe uma alternativa com licença permissiva: o **[LibreYOLO](/)** usa a licença MIT e é gratuito para qualquer uso comercial, sem restrições.

Se você pesquisou "is YOLOv8 free for commercial use", "YOLO license" ou "YOLO without AGPL", esta página responde com precisão, versão por versão.

## Licenças do YOLO em resumo

| Modelo | Licença padrão | Grátis para uso comercial de **código fechado**? | O que você precisa fazer |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | Não | Abrir o código-fonte de todo o aplicativo ou comprar uma Enterprise License |
| YOLOv8 | AGPL-3.0 | Não | Abrir o código-fonte de todo o aplicativo ou comprar uma Enterprise License |
| YOLO11 | AGPL-3.0 | Não | Abrir o código-fonte de todo o aplicativo ou comprar uma Enterprise License |
| YOLO26 | AGPL-3.0 | Não | Abrir o código-fonte de todo o aplicativo ou comprar uma Enterprise License |
| **LibreYOLO** | **MIT** | **Sim** | **Nada. Use em produtos proprietários, gratuitamente.** |

Em uma frase: os principais modelos YOLO são de código aberto, mas usam **AGPL-3.0**, uma licença copyleft que a maioria das empresas não consegue cumprir em um produto proprietário. O LibreYOLO oferece o mesmo fluxo de trabalho do YOLO sob a **licença MIT**, que não impõe essa obrigação.

## Qual é a licença do YOLO?

Os modelos YOLO mais usados, **YOLOv5, YOLOv8, YOLO11 e o mais recente YOLO26**, são distribuídos pelo fornecedor sob a **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 é uma licença de *copyleft forte*. Não é o tipo permissivo de licença de código aberto que diz "faça o que quiser", como MIT ou Apache-2.0. Ela traz uma obrigação específica e abrangente, e é justamente essa obrigação que cria problemas para quem usa o software comercialmente.

## O YOLOv8 é grátis para uso comercial?

Não da forma que a maioria das pessoas entende. Você *pode* baixar o YOLOv8 e usá-lo comercialmente, mas somente se cumprir a AGPL-3.0. Na prática, isso significa:

- Se você distribuir, entregar ou hospedar um produto que inclua YOLOv8, precisa **publicar o código-fonte completo desse produto**, incluindo seu aplicativo, o código de integração, os scripts e a configuração, sob AGPL-3.0.
- Essa obrigação é acionada não apenas quando você entrega software aos usuários, mas também quando permite que eles interajam com ele **pela rede**, conforme a cláusula da Seção 13, característica da AGPL. Um SaaS ou backend de API não fica isento.

Para um projeto pessoal ou uma pesquisa acadêmica, isso costuma ser aceitável. Para um produto comercial proprietário, publicar toda a base de código quase sempre é inviável. Tudo isso também se aplica ao **YOLO11** e ao **YOLO26**: mesma licença, mesma obrigação.

## E o YOLOv5, YOLO11 e YOLO26?

É a mesma situação. Todos usam **AGPL-3.0** por padrão. A Enterprise License do fornecedor menciona explicitamente "YOLO26, versões anteriores do YOLO e quaisquer modelos YOLO futuros", o que mostra que o licenciamento é intencional e consistente em toda a família. Escolher uma versão mais recente não garante uma licença mais favorável.

Para esclarecer: nem todo modelo com "YOLO" no nome usa AGPL. Algumas variantes da comunidade usam GPL-3.0 ou Apache-2.0, e as licenças variam conforme o autor. Mas as versões que as pessoas querem dizer quando pesquisam "YOLOv8" ou "YOLO11" usam AGPL-3.0.

## O que a AGPL-3.0 exige, em linguagem simples

A AGPL-3.0 estende a GPL com um acréscimo crucial: a **cláusula de uso em rede**. Na prática, é preciso:

1. **Distribuir o código-fonte.** Se você distribuir software que inclui código AGPL, precisa disponibilizar, sob AGPL-3.0, o *código-fonte correspondente completo* de toda a obra combinada.
2. **O uso pela rede também conta.** Se usuários interagirem com sua versão modificada pela rede, por meio de um aplicativo web, uma API ou um SaaS, você precisa oferecer a eles o mesmo código-fonte. Não existe uma brecha para "executamos tudo apenas nos nossos servidores".
3. **O copyleft se estende ao trabalho derivado.** A obrigação alcança a *obra derivada inteira*, não apenas os arquivos YOLO. Sua lógica de negócios também passa a estar sujeita à AGPL.

Isto é informação geral, não aconselhamento jurídico. Consulte um advogado para analisar seu caso específico. Mas a conclusão é simples: **AGPL-3.0 e software comercial de código fechado não combinam.**

## A opção da Enterprise License e sua ressalva

O fornecedor oferece uma **Enterprise License** paga que remove as obrigações da AGPL e permite incorporar YOLO a um produto proprietário sem abrir o código-fonte. É uma opção legítima e, para algumas empresas, é a escolha certa.

As ressalvas:

- **Custa dinheiro**, uma taxa comercial recorrente, negociada por empresa e com preços não divulgados.
- **Depende dos termos de um único fornecedor**, que podem mudar e cobrem apenas o código desse fornecedor.
- **Você está pagando para usar** uma arquitetura de modelo que, em sua essência, foi publicada como pesquisa.

Se uma taxa recorrente e a dependência de um fornecedor forem aceitáveis, a Enterprise License funciona. Se você preferir não pagar ou não assumir essas obrigações, continue lendo.

## A alternativa MIT: LibreYOLO

Divulgação: o LibreYOLO é nosso. É também por isso que esta página pode terminar com uma resposta direta, em vez de ficar em cima do muro.

**O LibreYOLO é um framework YOLO distribuído sob a licença permissiva MIT.** Esse único fato muda tudo para o uso comercial:

- **Gratuito para uso comercial**, inclusive em produtos proprietários e de código fechado.
- **Sem copyleft nem cláusula de uso em rede.** Você nunca precisa abrir o código-fonte do seu aplicativo.
- **Sem taxa por usuário nem taxa empresarial.** MIT significa MIT.
- **O mesmo fluxo de trabalho que você já conhece**, então a migração é simples.

O LibreYOLO é um framework real e mantido, não um wrapper: detecção de objetos, segmentação, pose, classificação, profundidade e muito mais, em uma API familiar, com treinamento e exportação para ONNX/TensorRT/OpenVINO/NCNN integrados. A diferença não está no fluxo de trabalho. Está na licença.

Se você chegou aqui por causa da questão mais ampla de "deixar o YOLO sob AGPL", escrevemos uma comparação mais completa do ecossistema em [Melhores alternativas ao Ultralytics em 2026](/articles/best-ultralytics-alternatives).

Para consultar a licença de todos os outros modelos YOLO, desde os originais Darknet em domínio público até YOLOv9, YOLOv10, YOLOv12, YOLOv13 e YOLO26, veja [Todas as licenças YOLO explicadas](/articles/yolo-licenses-explained).

### Por que a licença MIT importa para uma empresa

A licença MIT permite usar, modificar, incorporar e vender software baseado no LibreYOLO **sem obrigação de divulgar seu código-fonte** e **sem pagar nenhuma taxa**. É a licença usada em boa parte da pilha de software moderna justamente por ser segura para adoção comercial. O produto é seu, e você não deve nada.

## Experimente

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

Você mantém a experiência de desenvolvimento com YOLO: treine com seus dados, faça predições e exporte para distribuir. Você deixa para trás a obrigação da AGPL e a fatura da licença empresarial.

O LibreYOLO usa a licença MIT, roda em Linux, Mac e Windows e funciona em GPU, Apple Silicon e CPU comum sem nenhuma alteração no código. Uma API reúne YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, a linha RT-DETR, segmentação, pose, profundidade e muito mais, tudo com licenças permissivas.

Dê uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
