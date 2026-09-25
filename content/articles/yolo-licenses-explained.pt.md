---
title: "Licenças YOLO: guia completo da v1 à YOLO26 (2026)"
description: "Guia completo das licenças YOLO em 2026: de YOLOv1 a YOLOv13, YOLO26, YOLO-World e YOLOE, repositórios originais e reescritas permissivas, com links para artigos e GitHub, e o que você pode distribuir em um produto comercial."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9 é grátis para uso comercial?"
    a: "Depende do repositório usado. O repositório do artigo (WongKinYiu/yolov9) usa GPL-3.0. O mesmo laboratório também publica uma implementação separada de YOLOv9 e YOLOv7 sob MIT em MultimediaTechLab/YOLO, escrita depois que usuários comerciais pediram uma opção permissiva. É uma reescrita, não uma relicença dos arquivos GPL. A licença do código é clara; nenhum dos repositórios tem uma declaração de licença separada para os pesos pré-treinados, então trate a origem dos pesos como uma questão à parte."
  - q: "Quais versões de YOLO são grátis para uso comercial com código fechado?"
    a: "Por meio de pelo menos um repositório permissivo: YOLOv1 a YOLOv4 (Darknet original, domínio público), YOLOv7 e YOLOv9 (reescrita do próprio laboratório sob MIT), YOLOX e PP-YOLOE (Apache-2.0), além das reimplementações sob MIT no LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World e YOLOE não tinham uma implementação permissiva em julho de 2026. Confira o arquivo LICENSE atual antes de confiar nisso: as licenças mudam, e estas são informações gerais, não aconselhamento jurídico."
  - q: "Qual é a licença do YOLOv10?"
    a: "AGPL-3.0. YOLOv10 é uma publicação acadêmica da Universidade de Tsinghua, mas seu código foi construído sobre a base de código do Ultralytics, por isso herda AGPL-3.0. Não há uma reimplementação permissiva."
  - q: "Quais são as licenças do YOLOv12 e do YOLOv13?"
    a: "Ambos usam AGPL-3.0, conforme confirmado nos arquivos LICENSE. Assim como YOLOv10, são artigos acadêmicos cujo código de referência foi construído sobre o repositório do Ultralytics, então a AGPL se aplica independentemente de quem escreveu o artigo. Páginas de terceiros que listam YOLOv13 como Apache-2.0 estão erradas."
  - q: "Quando o Ultralytics mudou a licença do YOLO para AGPL-3.0?"
    a: "Em 14 de abril de 2023, não em 2022, como muita gente repete. O arquivo LICENSE em ultralytics/yolov5 só foi alterado três vezes, e o commit de AGPL (34cf749, PR #11359) é de 2023-04-14. A última versão de 2022, YOLOv5 v7.0, ainda foi publicada sob GPL-3.0. O repositório ultralytics/ultralytics recebeu a mesma relicença no mesmo dia. Portanto, YOLOv8 foi lançado em janeiro de 2023 sob GPL-3.0: as versões 8.0.0 a 8.0.76 no PyPI declaram GPL-3.0, e 8.0.80 (16 de abril de 2023) é a primeira a declarar AGPL-3.0."
  - q: "YOLO26 é grátis para uso comercial?"
    a: "Não em produtos de código fechado. YOLO26 é publicado sob AGPL-3.0, com uma Enterprise License paga como alternativa, nos mesmos termos de YOLOv8 e YOLO11."
  - q: "O YOLO original do Darknet está mesmo em domínio público?"
    a: "Sim. O arquivo LICENSE do Darknet, de Joseph Redmon, declara 'Darknet is public domain. Do whatever you want with it.' O fork de YOLOv4 de AlexeyAB traz o mesmo texto de domínio público. A ressalva é que os ports de PyTorch usados na prática têm suas próprias licenças, de Apache-2.0 a AGPL-3.0 ou nenhuma licença."
  - q: "Posso usar comercialmente os pesos pré-treinados do YOLO-NAS?"
    a: "Não. O código de super-gradients usa Apache-2.0, mas os pesos oficiais do YOLO-NAS são publicados sob uma licença separada que diz que você 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "Os pesos de uma rede neural herdam a licença do código de treinamento?"
    a: "A questão não está resolvida. O Ultralytics afirma que AGPL-3.0 cobre 'the training code and the models produced by that training code' e, como titular dos direitos autorais, define os termos do que publica. Nunca foi testado se um tribunal concordaria que os pesos que você treina são uma obra derivada do código de treinamento. Trate pesos publicados sob AGPL como sujeitos a essa licença e tenha cuidado ao carregá-los em uma reimplementação permissiva."
  - q: "Por que implementações diferentes do mesmo YOLO podem ter licenças diferentes?"
    a: "Os direitos autorais protegem a expressão, não as ideias (17 U.S.C. 102(b)). Uma arquitetura descrita em um artigo pode ser implementada de forma independente e licenciada como seu autor preferir. O que você não pode fazer é copiar arquivos-fonte GPL ou AGPL e relicenciá-los. Observe que esse argumento vale apenas para direitos autorais: uma implementação independente não é uma defesa contra patentes."
---

A cada ano surgem mais YOLOs, e a cada ano a pergunta sobre licenciamento aparece de novo, geralmente cinco minutos antes de uma decisão sobre o produto. A confusão vem do fato de que "YOLO" não é um projeto só. É uma marca compartilhada por cerca de vinte famílias de modelos de autores diferentes, e aqui está o detalhe que a maioria dos guias de licenças ignora: **a licença pertence a um repositório, não a um modelo.** A mesma versão do YOLO pode existir ao mesmo tempo em um repositório GPL e em um repositório MIT, dos mesmos autores. É o caso de YOLOv9, e isso muda toda a decisão.

Este guia mapeia o cenário repositório por repositório, com links para cada artigo e base de código, atualizado até julho de 2026. Conferimos cada licença abaixo no arquivo LICENSE do próprio repositório, porque uma quantidade surpreendente do que se escreve sobre as licenças do YOLO (inclusive nos guias mais bem posicionados para essa busca) está errada.

Se você só quer a resposta sobre a AGPL do Ultralytics, explicamos isso em detalhes em [YOLO é grátis para uso comercial?](/articles/yolo-commercial-license). Este artigo é o mapa completo.

**Duas observações antes de começar.** Primeiro, mantemos o LibreYOLO, uma biblioteca sob MIT que concorre com vários projetos abaixo, e a última seção deste artigo é sobre ela. Esse é um motivo para conferir o que escrevemos, não para aceitar sem questionar, por isso cada afirmação sobre licença aqui aponta para a fonte primária. Segundo, este artigo traz informações gerais sobre textos de licenças publicados e declarações públicas, válidas na data indicada acima. Não é aconselhamento jurídico nem substitui a orientação de um advogado na sua jurisdição. As licenças e as posições dos mantenedores mudam. Leia o arquivo LICENSE atual de qualquer repositório antes de decidir o que vai distribuir.

## A tabela de licenças

"Opção permissiva" significa: um repositório que implementa esse modelo e que você pode usar em um produto comercial de código fechado, sem abrir o código da sua aplicação e sem pagar por uma licença.

| Modelo | Ano | Código original | Opção permissiva | Artigo |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domínio público) | O próprio original | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domínio público) | O próprio original | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domínio público) | O original; atenção ao port de PyTorch sob AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (domínio público) | O original; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Nenhuma | sem artigo |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, mesmo laboratório) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 no lançamento, veja abaixo) | Nenhuma (o port de Keras foi descontinuado, veja abaixo) | sem artigo |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, mesmo laboratório) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Nenhuma | sem artigo |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | O próprio original | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | O original, do PaddleDetection (não PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (código Apache-2.0, pesos não comerciais) | Código, sim; pesos oficiais, não | sem artigo |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Nenhuma | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, sem atividade desde 2024) | Nenhuma | sem artigo |

Leia a tabela coluna por coluna e uma coisa fica óbvia: o número da versão não diz nada sobre a licença. O repositório, sim.

## Mesmo modelo, duas licenças: o caso do YOLOv9

YOLOv9 é a prova mais clara de que perguntar "qual é a licença do YOLOvN?" é fazer a pergunta errada. Vale contar a cronologia com precisão porque esta é a única vez na história do YOLO em que a pressão da comunidade realmente mudou o resultado.

- **18 de fevereiro de 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) é publicado junto com o [artigo](https://arxiv.org/abs/2402.13616).
- **22 de fevereiro de 2024:** um usuário abre a issue #10 para perguntar qual é a licença. Chien-Yao Wang (WongKinYiu) responde "I think it should be GPL3" e adiciona o arquivo GPL-3.0 quatro dias depois.
- **26 de fevereiro de 2024:** outro usuário abre a [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Nas duas semanas e meia seguintes, uma dúzia de usuários comerciais participa da discussão.
- **14 de março de 2024:** Wang publica nesta discussão: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

Esse repositório, chamado inicialmente `yolov9mit`, hoje é o **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: usa MIT, tem direitos autorais de "Kin-Yiu, Wong and Hao-Tang, Tsui" e se descreve como *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* A maior parte da reescrita não foi feita pelos voluntários da discussão, mas por Hao-Tang Tsui, colega de laboratório de Wang, autor de cerca de 90 por cento dos commits.

Assim, YOLOv9 é ao mesmo tempo "não distribuível" e "distribuível", dependendo do repositório que você clona. Mesma arquitetura, mesmo laboratório, duas licenças.

**Antes de apostar um produto nisso, considere três ressalvas que o repositório não informa na página inicial:**

1. **Ainda está amadurecendo.** Uma [issue aberta](https://github.com/MultimediaTechLab/YOLO/issues/231) de fevereiro de 2026 relata que os checkpoints publicados não reproduzem os números de COCO do artigo e têm bem mais parâmetros do que o artigo afirma. O treinamento de segmentação e keypoints [ainda não foi implementado](https://github.com/MultimediaTechLab/YOLO/issues/232). Nada foi integrado à branch main desde a tag v1.0, em dezembro de 2025.
2. **Os pesos não têm uma declaração de licença separada.** Pela convenção, a licença MIT do repositório cobre os assets da versão publicada, e as evidências técnicas apontam para checkpoints treinados por este projeto, não copiados do repositório GPL: os arquivos diferem em tamanho e número de parâmetros dos checkpoints do repositório GPL. Mas nenhum documento declara que os arquivos .pt usam MIT, e um [pedido da comunidade para uma auditoria formal contra contaminação por AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) continua aberto. Falta documentação: ninguém afirmou que os arquivos foram "verificados como limpos", e ninguém apontou um problema.
3. **É outra base de código**, não um substituto direto para os scripts do repositório GPL.

Nada disso torna o projeto inutilizável. Significa que você deve avaliá-lo, não apenas marcar uma caixa.

## Era 1: os anos de domínio público (YOLOv1 a YOLOv4)

Joseph Redmon publicou [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (publicado como o artigo YOLO9000) e [YOLOv3](https://arxiv.org/abs/1804.02767) dentro do framework Darknet. O [arquivo LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) é famoso por ter apenas três linhas:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

É uma dedicação genuína ao domínio público. [YOLOv4](https://arxiv.org/abs/2004.10934), mantido por Alexey Bochkovskiy em um [fork do Darknet](https://github.com/AlexeyAB/darknet), traz o *mesmo* arquivo, não a Unlicense que vários guias de licenças atribuem a ele. O classificador do próprio GitHub não consegue identificá-lo e informa "Other". Vale saber disso se sua ferramenta de conformidade lê esse campo: um scanner vai sinalizá-lo como licença não identificada, em vez de classificá-lo como uma licença permissiva, embora o texto dificilmente pudesse ser mais permissivo.

O próprio Darknet também não está exatamente morto. O repositório de Redmon está parado desde 2022, mas o README de AlexeyAB agora indica [hank-ai/darknet](https://github.com/hank-ai/darknet) como sucessor recomendado em C/C++, mantido ativamente.

**A armadilha são os ports.** Em 2026, quase ninguém treina com Darknet; as pessoas recorrem a uma reimplementação de PyTorch, que tem sua própria licença:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), o port de YOLOv3 mais popular, usa **AGPL-3.0**. Uma arquitetura em domínio público redistribuída sob a licença copyleft mais rigorosa comum neste cenário. A licença que você recebe depende inteiramente de qual port instalou com pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) usa **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) usa **Apache-2.0**, então ainda há uma opção permissiva para v4 em PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), do coautor de YOLOv4, não tem **nenhum arquivo LICENSE**. Isso é pior que copyleft: sem licença, nenhum direito é concedido, e o padrão é todos os direitos reservados.

Confira a licença do repositório que você clonou, não o número da versão no artigo.

## Era 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7) e [YOLOv9](https://github.com/WongKinYiu/yolov9) foram publicados sob GPL-3.0.

GPL-3.0 é copyleft: se você distribuir uma versão modificada ou combinar código GPL com o seu em um único programa, deve fornecer o código-fonte correspondente completo desse trabalho combinado sob GPL-3.0. Há dois limites importantes. **Mera agregação** significa que programas separados e independentes, enviados juntos, não são automaticamente considerados combinados. E o gatilho é a **distribuição**: ao contrário da AGPL, se você nunca distribuir o trabalho combinado, a GPL simples não impõe obrigação de divulgação. Por isso algumas empresas mantêm um modelo GPL estritamente por trás da própria API.

Esse caminho é mais estreito do que parece e falha de três maneiras que pouca gente planeja. Quando o modelo chega ao dispositivo de um cliente ou a uma instalação local, você está distribuindo. "Só para uso interno" deixa de ser verdade quando uma cópia sai da sua pessoa jurídica: entregar uma versão compilada a um prestador de serviços, uma equipe terceirizada de desenvolvimento ou uma afiliada constituída separadamente é distribuição, embora copiar o arquivo dentro da mesma empresa não seja. E essa proteção só funciona se *tudo* que pode ser acessado por essa API usar GPL ou uma licença mais permissiva, pois um único componente AGPL em qualquer ponto do trabalho servido submete o conjunto à Seção 13, independentemente do que diga o próprio arquivo GPL.

**YOLOv7 e YOLOv9 têm a alternativa MIT descrita acima.** YOLOv6 não tem, e a evidência mais forte disso vem da Baidu: PaddleDetection usa Apache-2.0, e seus mantenedores isolaram deliberadamente YOLOv5, YOLOv6, YOLOv7 e YOLOv8 em um repositório GPL-3.0 separado, o [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), afirmando que o código desses modelos "will not be merged into PaddleDetection." Quando uma empresa do porte da Baidu cria uma barreira para manter YOLOv6 fora do seu framework permissivo, isso mostra o valor da licença.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

O Ultralytics mudou para AGPL-3.0 em **14 de abril de 2023**, não em 2022, como é amplamente repetido, inclusive em guias que aparecem na primeira página dos resultados desta busca. Tudo que veio depois ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11 e [YOLO26](https://arxiv.org/abs/2606.03748)) usa AGPL-3.0, com uma Enterprise License paga como alternativa. A [página de licenças](https://ultralytics.com/license) deles diz que o nível Enterprise cobre "YOLO26, earlier YOLO versions, and any future YOLO models", então a política é deliberada e voltada para o futuro. Os preços não são públicos.

Essa data pode ser conferida e vale a pena verificar, pois a versão popular dessa história está errada de uma forma que faz diferença:

- O arquivo `LICENSE` em `ultralytics/yolov5` foi alterado exatamente três vezes ao longo de sua história. A terceira alteração é o commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)", datado de **2023-04-14**, que substitui `GNU GENERAL PUBLIC LICENSE` por `GNU AFFERO GENERAL PUBLIC LICENSE` em 101 arquivos.
- A última versão do YOLOv5 de 2022, **v7.0 (22 de novembro de 2022), ainda foi publicada sob GPL-3.0**, assim como v6.2, v6.1 e v6.0. O mesmo vale para os milhares de forks que pararam de sincronizar antes de abril de 2023: escolha qualquer um deles e o GitHub ainda informa GPL-3.0 hoje.
- No mesmo dia, 41 minutos depois, `ultralytics/ultralytics` recebeu [a mesma alteração](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Isso revela um fato que quase ninguém conhece: **YOLOv8 não foi lançado sob AGPL.** Foi publicado em janeiro de 2023 sob **GPL-3.0** e relicenciado três meses depois. O registro do PyPI não deixa dúvidas: `ultralytics` 8.0.0 (10 de janeiro de 2023) a 8.0.76 (13 de abril de 2023) declaram GPL-3.0. A versão 8.0.80, enviada em 16 de abril de 2023, é a primeira a declarar AGPL-3.0.

Isso não é uma brecha. Uma licença concedida não é revogada retroativamente, então essas versões antigas continuam disponíveis nos termos em que foram publicadas. Mas estão três anos defasadas, sem manutenção nem correções; GPL-3.0 continua sendo copyleft (você trocou uma cláusula de rede por uma cláusula de distribuição, não escapou do copyleft), e a posição comercial do Ultralytics é mais abrangente que o texto da licença. A conclusão útil é mais específica e geral: **a licença se aplica à versão que você recebeu, não ao nome do projeto.** Fixe suas dependências e registre o que a licença dizia no dia em que você obteve a cópia, porque o projeto pode mudar isso amanhã e seu histórico de conformidade depende da versão que você realmente recebeu.

AGPL-3.0 é GPL com a Seção 13, a cláusula de rede: *se você modificar o Programa*, os usuários que interagem com sua versão modificada pela rede devem receber a oferta do código-fonte. Isso fecha o caminho de SaaS que a GPL simples deixa aberto.

Observe a ressalva, que a maioria dos artigos omite. A Seção 0 da AGPL define "modificar" como copiar ou adaptar o trabalho *de uma forma que exija autorização de direitos autorais*. Executar código de treinamento sem modificações com seu próprio dataset é executar o programa, não adaptar seu código-fonte, assim como compilar seu código com um compilador GPL sem modificações não altera o compilador. Portanto, o treinamento por si só não aciona automaticamente a obrigação, e quem afirma isso ignorou a definição.

O que coloca você no escopo da licença é combinar ou incorporar o pacote à sua própria base de código, de modo que os dois sejam distribuídos ou servidos como um único programa. A interpretação da própria FSF é que vincular um trabalho GPL ou AGPL a outros módulos cria um trabalho combinado; nunca foi testado se um tribunal estenderia isso a ponto de incluir um `import` em Python. Observe também que os [termos comerciais](https://ultralytics.com/license) do próprio Ultralytics exigem uma Enterprise License para plataformas SaaS, APIs e sistemas em nuvem que usam YOLO nos bastidores, então a posição do fornecedor é mais ampla que o texto básico da licença. Considere a integração a um produto que você distribui ou serve como o risco; não presuma que treinar isoladamente seja suficiente.

Especificamente sobre YOLO26, as datas aparecem confusas em todos os lugares, então: ele foi **apresentado em setembro de 2025** no YOLO Vision, **lançado de fato em janeiro de 2026** (na versão 8.4.0 de `ultralytics`, cujas notas dizem "Ultralytics YOLO26 has arrived") e o **artigo saiu em junho de 2026**. Ele está no repositório principal `ultralytics/ultralytics`; `ultralytics/yolo26` é apenas uma página de apresentação.

**O que surpreende as pessoas:** os YOLOs acadêmicos dos últimos dois anos também usam AGPL, e não porque o Ultralytics os escreveu.

- [YOLOv10](https://arxiv.org/abs/2405.14458) é da Universidade de Tsinghua. O README diz *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), o modelo de vocabulário aberto "veja qualquer coisa" do grupo YOLOv10, também usa AGPL-3.0 e foi construído sobre o Ultralytics.

O copyleft é herdado: um derivado de código AGPL precisa ser distribuído sob AGPL, então, quando um desses forks é distribuído ou servido, a licença vem junto. (Se você modificar o código e mantiver o resultado estritamente para uso próprio, nenhuma obrigação se aplica, por isso o uso em pesquisa não é afetado.) A pesquisa é independente; a licença, não. Desde 2024, publicar "o próximo YOLO" como derivado do Ultralytics virou o fluxo de trabalho padrão, o que faz a AGPL se propagar automaticamente para as versões seguintes.

Se você encontrar uma página dizendo que YOLOv13 usa Apache-2.0, ela está errada. O arquivo LICENSE, a API do GitHub e o cartão do modelo dizem AGPL-3.0.

### A opção permissiva para YOLOv8 que deixou de existir

Muitos guias ainda disponíveis, inclusive uma versão anterior deste artigo, apontam para o YOLOv8 do KerasCV sob Apache-2.0 como uma alternativa segura ao YOLOv8 sob AGPL. **Hoje não dá para contar com essa opção.** Este é o registro público:

- Na discussão do KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), um colaborador escreveu que "many parts are derived form ultralytics." Nenhum mantenedor respondeu à discussão, então a questão sobre o grau de independência da implementação continua sem resposta pública.
- A issue do KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) foi encerrada em julho de 2025 por um mantenedor do Keras, que disse: "Hey all!! Because of license issues, we have decided to drop this." Depois, duas pessoas perguntaram o que isso significava para quem já usava comercialmente a versão do KerasCV. Nenhuma recebeu resposta.
- O KerasCV foi arquivado, e o KerasHub, seu sucessor mantido, **não distribui YOLOv8**.

Não sabemos se essas duas coisas estão relacionadas, e não estamos afirmando que houve algo impróprio. Isso não é uma base sólida para construir um produto.

## As exceções permissivas: YOLOX, PP-YOLOE, YOLO-NAS

Nem todo YOLO entrou na onda do copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) usa Apache-2.0 sem exceções, tanto para o código quanto para os pesos. Ainda é o YOLO clássico mais simples para uso comercial, embora o repositório não receba commits desde meados de 2025, e uma [pergunta sobre incluir os pesos pré-treinados em um aplicativo comercial](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) esteja aberta desde março de 2026, sem resposta. O texto da licença não deixa dúvidas. Só não há ninguém por perto para responder às perguntas. Escrevemos sobre [executar YOLOX com LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) usa Apache-2.0 **dentro do PaddleDetection**. Obtenha-o dali, não do PaddleYOLO, que tem um diretório de configuração quase idêntico e usa GPL-3.0. Mesmo modelo, mesma empresa, dois repositórios, duas licenças. É o caso do YOLOv9 ao contrário.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) é o que pega muita gente de surpresa. O código usa Apache-2.0; os pesos oficiais, não. A [licença separada](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) declara que você *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Desde que a NVIDIA adquiriu a Deci em 2024, não encontramos novos trabalhos de desenvolvimento de funcionalidades no repositório, e os URLs originais dos pesos não funcionam mais (os checkpoints agora estão em outro host). Às vezes sugerem como alternativa que treinar a arquitetura do zero mantém tudo sob Apache, o que é plausível, mas não foi confirmado pela Deci nem pela NVIDIA e não combina bem com o texto da licença, que inclui "any components comprising the model". Saiba mais sobre YOLO-NAS [aqui](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** às vezes aparece como Apache-2.0 em guias antigos. Não é: MMYOLO usa GPL-3.0 (sua biblioteca irmã MMDetection é a que usa Apache), e não recebe commits desde julho de 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) usa **GPL-3.0**, não AGPL nem uma licença permissiva. Vale dizer isso porque muitas vezes se presume que YOLOs de vocabulário aberto são permissivos por associação ao ecossistema no estilo CLIP de que aproveitam recursos.

Esses casos cobrem os três erros de licenciamento mais comuns neste cenário: presumir que os pesos seguem a licença do código, presumir que tudo de uma organização usa a mesma licença e presumir a licença de um modelo pelo nome.

## Licenças cobrem código, não ideias (mas atenção às patentes)

Um princípio é a base jurídica de todas as opções permissivas acima: **os direitos autorais protegem a expressão, não as ideias.** Isso não é um slogan, é lei consolidada nos dois lados do Atlântico.

Nos EUA, essa regra está em [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), que exclui da proteção autoral qualquer "idea, procedure, process, system, method of operation, concept, principle, or discovery", desde *Baker v. Selden* (1879). Na UE, cuja lei se aplica a nós e provavelmente a muitos leitores, o Artigo 1(2) da [Diretiva de Software (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) diz que "ideas and principles which underlie any element of a computer program" não são protegidos, e o Tribunal de Justiça confirmou em *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) que a funcionalidade de um programa, sua linguagem de programação e seus formatos de dados não são, por si só, expressão protegida. Só o código é.

Uma arquitetura descrita em um artigo do arXiv é uma ideia publicada. Qualquer pessoa pode implementá-la. O que você não pode fazer é copiar ou adaptar arquivos-fonte GPL/AGPL de outra pessoa e relicenciar o resultado. Uma implementação independente, escrita por pessoas que consultaram o artigo e não o código-fonte, é um trabalho novo, cuja licença fica a critério de quem o criou.

Duas ressalvas importantes que a maioria dos fornecedores omite:

- **A independência precisa ser real.** A versão defensável disso é um processo de clean room: quem escreve o código trabalha a partir do artigo e de uma especificação, não do código-fonte copyleft. Ler o repositório GPL com atenção e depois "reescrevê-lo" não é a mesma coisa, e essa diferença importa se alguém investigar.
- **Este é um argumento sobre direitos autorais, não sobre patentes.** A invenção independente *não* é uma defesa contra uma patente. Uma reimplementação clean room resolve a questão dos direitos autorais e deixa a questão das patentes exatamente como estava. Hoje não há uma patente amplamente reivindicada sobre as arquiteturas YOLO, mas "ninguém processou ainda" não significa "não há nada que possa motivar um processo".

## Os pesos têm uma segunda licença

O erro mais caro neste campo é auditar o código e esquecer o checkpoint. Três camadas têm licenças, e elas podem ser diferentes:

**1. O código.** Tudo que vimos acima.

**2. Os pesos.** Às vezes a licença é explícita, como no YOLO-NAS. Às vezes é declarada: a [página de licenças](https://ultralytics.com/license) do Ultralytics afirma que AGPL-3.0 "covers the training code and the models produced by that training code" e estende isso a modelos que você mesmo treina com o código deles. Como titular dos direitos autorais do que publica, o projeto pode definir os termos dos próprios checkpoints. A questão de saber se os pesos que *você* produz são legalmente uma obra derivada do código de treinamento continua realmente em aberto: não há jurisprudência, e isso contraria a regra geral da própria FSF de que a saída de um programa não é automaticamente coberta pelos direitos autorais do programa. Mas uma questão em aberto não significa que seja seguro. Na prática, trate os pesos publicados sob AGPL como sujeitos a essa licença.

A ressalva importante para quem seguir a recomendação deste artigo: **os repositórios GPL de YOLOv7 e YOLOv9 não dizem absolutamente nada sobre os pesos.** Os checkpoints são assets de versões publicadas em repositórios GPL-3.0 sem uma concessão separada. Carregar esses arquivos .pt em uma reimplementação MIT deixa você com código limpo, mas com um checkpoint cuja situação ninguém esclareceu. Se o motivo da mudança foi a licença, use pesos treinados pelo projeto permissivo ou treine os seus próprios.

**3. Os dados de treinamento.** As *anotações* de COCO usam CC BY 4.0, então os rótulos estão em ordem. As *imagens* não são licenciadas pelo próprio COCO: são fotos do Flickr sujeitas a um conjunto variado de termos individuais, que o consórcio COCO declara explicitamente não possuir. A maior parte do setor considera esse risco baixo e segue em frente, mas dizer "COCO está liberado" se refere apenas às anotações. Fora do COCO, as restrições aumentam rapidamente: muitos datasets aéreos, médicos e de direção não permitem uso comercial, e a licença permissiva do código de um modelo não regulariza isso.

Audite as três camadas sempre.

## Uma observação sobre MIT e Apache-2.0

Publicamos uma biblioteca sob MIT, então seria conveniente dizer que MIT é simplesmente a melhor licença. A comparação honesta é mais interessante.

MIT não significa ausência de obrigações: você precisa manter o aviso de direitos autorais e o texto da licença nas cópias que distribuir. É uma condição real, mas simples de cumprir. E Apache-2.0 tem algo que MIT não tem: uma **concessão expressa de patentes** de cada colaborador, além de uma cláusula de retaliação que encerra a concessão de quem processar por causa do trabalho. MIT não trata de patentes. Para uma empresa cujo maior receio é a exposição a patentes, Apache-2.0 pode ser a opção *mais segura*, e boa parte do ecossistema permissivo de detecção (YOLOX, RT-DETR, D-FINE, DEIM) usa Apache-2.0 exatamente por esse motivo.

O que MIT oferece é simplicidade e compatibilidade. Nenhuma dessas licenças vai obrigar você a publicar seu código-fonte, que é o critério central deste artigo. Se alguém disser que escolher entre MIT e Apache é a decisão importante, está tentando vender algo. A decisão importante é entre permissiva e copyleft.

## Então, qual YOLO você pode usar de fato?

- **Produto comercial de código fechado:** YOLOv7 ou YOLOv9 pelo [repositório MIT](https://github.com/MultimediaTechLab/YOLO) do laboratório; YOLOX ou PP-YOLOE (do PaddleDetection) entre os originais; ou um framework MIT mantido, como LibreYOLO, se preferir não montar e auditar tudo por conta própria. Evite qualquer opção GPL ou AGPL, a menos que vá abrir toda a aplicação ou comprar a licença Enterprise.
- **Projeto de código aberto:** qualquer um, desde que a licença seja compatível. Se seu projeto usa AGPL, a linha do Ultralytics é uma opção natural, e as pesquisas mais recentes (YOLOv13, YOLO26, YOLOE) aparecem primeiro nela.
- **Pesquisa e benchmarking:** as licenças impõem poucas restrições. Use o que tiver sido usado no artigo com o qual você está comparando.
- **A estratégia "a gente vê isso depois":** não funciona. Remover uma dependência AGPL depois que o produto está pronto significa treinar de novo, revalidar e exportar tudo novamente, inclusive os pesos. Escolha primeiro o repositório e, portanto, a licença.

Para comparar os próprios frameworks, e não as licenças, veja [Melhores alternativas ao Ultralytics em 2026](/articles/best-ultralytics-alternatives).

## A opção MIT: o que o LibreYOLO oferece

Observação: o LibreYOLO é nosso projeto, e sua licença é o motivo da existência deste artigo.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) é uma única base de código sob MIT** que oferece os detectores abaixo por meio de uma API, com treinamento e exportação para ONNX, TensorRT, OpenVINO e NCNN integrados. Sem copyleft, sem cláusula de rede, sem nível Enterprise.

Esta é a lista completa de modelos de detecção de objetos, com o projeto upstream de cada família e sua licença, para você ver exatamente de onde vem cada opção permissiva:

| No LibreYOLO | Modelo | Upstream | Licença upstream |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | YOLOs da era Darknet, em PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Domínio público |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (reescrita do próprio laboratório, não o repositório GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9 e uma variante end-to-end sem NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (reescrita do próprio laboratório, não o repositório GPL) |
| `LibreYOLO9P2` | YOLOv9 com uma cabeça stride-4 para objetos pequenos | Original do LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, detecção e pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Código Apache-2.0. Os pesos upstream restritos **não** são redistribuídos |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR e v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, detecção, segmentação, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 para N/S/M/L. XL/2XL usam a Platform Model License da Roboflow, então distribuímos apenas os tamanhos Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Código Apache-2.0. Os tamanhos maiores usam o backbone DINOv3 da Meta sob uma licença própria da Meta, não aprovada pela OSI, que permite redistribuição, mas **proíbe uso militar, nuclear, de espionagem e de armas**. Esses tamanhos são publicados como `other`, não como Apache. Leia a licença antes de distribuir algo de uso dual |
| `LibreRTMDet` | RTMDet ([sem MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (a biblioteca irmã Apache, não o mmyolo sob GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Arquitetura do [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoints via o re-port [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (ambos) |
| `LibreEC` | EdgeCrafter, detecção, pose, segmentação | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Detector de centroides para hardware da classe de microcontroladores | Original do LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, vocabulário aberto (inferência) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, vocabulário aberto (inferência) | Google, via `transformers` | Apache-2.0 |

Há dois pontos sobre os quais essa coluna é deliberadamente transparente.

**Nenhum código-fonte GPL ou AGPL é copiado para a base de código.** Todas as famílias acima foram construídas a partir de código upstream em domínio público ou sob MIT ou Apache-2.0, regra que mantém o framework sob MIT. Quando uma arquitetura tem uma opção permissiva e outra copyleft, usamos a permissiva: YOLOv9 e YOLOv7 vêm do repositório MIT do laboratório, não dos originais GPL; e RTMDet vem do MMDetection sob Apache, não do MMYOLO sob GPL. YOLO-World não aparece na lista, embora seja pedido com frequência, porque usa GPL-3.0 e não há uma fonte permissiva de onde obtê-lo.

**Permissiva não significa sem restrições, e preferimos deixar isso claro a deixar você descobrir durante uma auditoria.** Os tamanhos maiores do DEIMv2 herdam os termos do DINOv3 da Meta, incluindo a proibição de uso militar, nuclear, de espionagem e de armas, uma restrição real se você trabalha com algo de uso dual. Alguns checkpoints de prévia de pesquisa foram treinados com datasets não comerciais (DOTA, VisDrone) e estão identificados como `cc-by-nc` no Hugging Face, em vez de serem publicados silenciosamente como se fossem livres. E a mesma cautela que aplicamos aos pesos de outras pessoas vale para os nossos: nossos checkpoints YOLO9 são convertidos a partir dos checkpoints do próprio repositório MIT do laboratório, então herdam o que quer que se aplique a eles, que, como observado acima, é "declarado, mas não auditado formalmente". A licença MIT cobre o código que escrevemos. Cada checkpoint traz os termos dos dados e do material a partir dos quais foi treinado, publicados para cada peso.

Além de detecção, a mesma API oferece segmentação de instâncias e semântica, pose, caixas orientadas, classificação (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), estimativa monocular de profundidade, restauração de imagem e estimativa do olhar.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

O mesmo fluxo de trabalho com YOLO, sem toda a pesquisa sobre licenças.

Adicione uma estrela no GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Documentação: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Marcas registradas e afiliação: YOLO, YOLOv8, YOLO11, YOLO26 e Ultralytics são marcas registradas de seus respectivos titulares, incluindo Ultralytics Inc. LibreYOLO é um projeto independente e não é afiliado, patrocinado nem endossado pelo Ultralytics ou por qualquer outra organização mencionada neste artigo. Os nomes de produtos e repositórios são usados apenas para identificar e comparar o software discutido.*

*Este artigo traz informações gerais, não aconselhamento jurídico, válidas em 11 de julho de 2026, e foi escrito por um dos fornecedores na comparação. Confira a licença atual antes de distribuir.*
