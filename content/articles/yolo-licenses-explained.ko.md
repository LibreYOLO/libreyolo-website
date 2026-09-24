---
title: "YOLO 라이선스 총정리: YOLOv1부터 YOLO26까지 (2026 가이드)"
description: "2026년 기준 YOLO 라이선스 지도 완전판. YOLOv1부터 YOLOv13, YOLO26, YOLO-World, YOLOE까지 원본 저장소와 허용적 라이선스 재작성본을 논문 및 GitHub 링크와 함께 정리하고, 상업 제품에 실제로 넣을 수 있는 것이 무엇인지 설명합니다."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9은 상업적으로 무료로 사용할 수 있습니까?"
    a: "어떤 저장소를 사용하느냐에 따라 다릅니다. 논문 저장소(WongKinYiu/yolov9)는 GPL-3.0입니다. 같은 연구실은 MultimediaTechLab/YOLO에서 YOLOv9과 YOLOv7의 MIT 라이선스 구현체를 별도로 제공하며, 이는 상업 사용자들이 허용적 라이선스 옵션을 요청한 뒤에 작성되었습니다. 이 구현체는 재작성본이며, GPL 파일의 라이선스를 바꾼 것이 아닙니다. 코드 라이선스는 명확합니다. 다만 사전 학습된 가중치에는 두 저장소 모두 별도의 라이선스 명시가 없으므로, 가중치의 출처는 별개의 문제로 다루어야 합니다."
  - q: "비공개 소스 상업적 사용이 무료인 YOLO 버전은 무엇입니까?"
    a: "허용적 라이선스 저장소가 적어도 하나 있는 버전은 다음과 같습니다. YOLOv1부터 YOLOv4(원본 Darknet, 퍼블릭 도메인), YOLOv7과 YOLOv9(연구실 자체의 MIT 재작성본), YOLOX와 PP-YOLOE(Apache-2.0), 그리고 LibreYOLO의 MIT 재구현체입니다. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World, YOLOE는 2026년 7월 기준으로 허용적 라이선스 구현체가 없습니다. 이 내용에 의존하기 전에 현재 LICENSE 파일을 직접 확인하십시오. 라이선스는 바뀌며, 이 내용은 일반적인 정보이지 법률 자문이 아닙니다."
  - q: "YOLOv10의 라이선스는 무엇입니까?"
    a: "AGPL-3.0입니다. YOLOv10은 칭화대학교(Tsinghua University)의 학술 릴리스이지만, 코드가 Ultralytics 코드베이스를 기반으로 만들어졌기 때문에 AGPL-3.0을 물려받습니다. 허용적 라이선스 재구현체는 없습니다."
  - q: "YOLOv12와 YOLOv13의 라이선스는 무엇입니까?"
    a: "둘 다 AGPL-3.0이며, 각 LICENSE 파일에서 확인됩니다. YOLOv10과 마찬가지로 참조 코드가 Ultralytics 저장소를 기반으로 만들어진 학술 논문이므로, 논문 저자가 누구든 AGPL이 그대로 이어집니다. YOLOv13을 Apache-2.0으로 표기한 제3자 페이지는 틀렸습니다."
  - q: "Ultralytics는 언제 YOLO를 AGPL-3.0으로 전환했습니까?"
    a: "널리 알려진 것처럼 2022년이 아니라 2023년 4월 14일입니다. ultralytics/yolov5의 LICENSE 파일은 지금까지 단 세 번 변경되었으며, AGPL 커밋(34cf749, PR #11359)의 날짜는 2023-04-14입니다. 2022년의 마지막 릴리스인 YOLOv5 v7.0은 여전히 GPL-3.0으로 배포되었습니다. ultralytics/ultralytics 저장소도 같은 날 라이선스가 변경되었습니다. 따라서 YOLOv8은 2023년 1월 GPL-3.0으로 출시되었습니다. PyPI 릴리스 8.0.0부터 8.0.76까지는 GPL-3.0을 선언하며, 8.0.80(2023년 4월 16일)이 AGPL-3.0을 선언한 첫 버전입니다."
  - q: "YOLO26 모델은 상업적으로 무료로 사용할 수 있습니까?"
    a: "비공개 소스 제품에는 사용할 수 없습니다. YOLO26 모델은 AGPL-3.0으로 배포되며 대안으로 유료 엔터프라이즈 라이선스가 있습니다. YOLOv8, YOLO11과 같은 조건입니다."
  - q: "원래의 Darknet YOLO는 정말 퍼블릭 도메인입니까?"
    a: "그렇습니다. Joseph Redmon의 Darknet LICENSE에는 'Darknet is public domain. Do whatever you want with it.'(Darknet은 퍼블릭 도메인이다. 마음대로 사용하라.)라고 적혀 있습니다. AlexeyAB의 YOLOv4 포크에도 같은 퍼블릭 도메인 문구가 있습니다. 함정은 사람들이 실제로 쓰는 PyTorch 포팅판이 각자 고유한 라이선스를 가진다는 점입니다. Apache-2.0부터 AGPL-3.0, 라이선스가 아예 없는 경우까지 다양합니다."
  - q: "YOLO-NAS의 사전 학습된 가중치를 상업적으로 사용할 수 있습니까?"
    a: "사용할 수 없습니다. super-gradients 코드는 Apache-2.0이지만, 공식 YOLO-NAS 가중치는 별도의 라이선스로 배포되며, 그 라이선스에는 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'(프로덕션 환경에서 사용되는 모든 모델과 관련된 사용을 포함해, 어떠한 상업적 용도로도 본 소프트웨어를 사용할 수 없다.)라고 적혀 있습니다."
  - q: "신경망 가중치는 학습 코드의 라이선스를 물려받습니까?"
    a: "아직 정리되지 않은 문제입니다. Ultralytics는 AGPL-3.0이 'the training code and the models produced by that training code,'(학습 코드와 그 학습 코드로 만들어진 모델)를 포괄한다고 밝히고 있으며, 저작권자로서 자신들이 공개하는 것의 조건은 스스로 정합니다. 직접 학습한 가중치가 학습 코드의 2차적 저작물이라는 데 법원이 동의할지는 한 번도 다투어진 적이 없습니다. 공개된 AGPL 가중치는 제약이 걸린 것으로 취급하고, 허용적 라이선스 재구현체에 불러올 때는 주의하십시오."
  - q: "같은 YOLO라도 구현체마다 라이선스가 다를 수 있는 이유는 무엇입니까?"
    a: "저작권은 아이디어가 아니라 표현을 보호합니다(17 U.S.C. 102(b)). 논문에 설명된 아키텍처는 독립적으로 구현할 수 있고, 그 구현체의 라이선스는 작성자가 원하는 대로 정할 수 있습니다. 할 수 없는 것은 GPL 또는 AGPL 소스 파일을 복사해 라이선스를 바꾸는 것입니다. 이는 저작권 논리일 뿐이라는 점에 유의하십시오. 독립적인 구현은 특허에 대한 방어 수단이 되지 않습니다."
---

해마다 YOLO는 늘어나고 라이선스 질문도 해마다 다시 나오는데, 대개 제품 결정을 5분 앞둔 시점입니다. 혼란스러운 점은 "YOLO"가 하나의 프로젝트가 아니라는 것입니다. 서로 다른 작성자가 만든 스무 개 남짓한 모델 계열이 함께 쓰는 브랜드 이름이며, 대부분의 라이선스 가이드가 놓치는 세부 사항은 이것입니다: **라이선스는 모델이 아니라 저장소에 속합니다.** 같은 YOLO 버전이 같은 작성자가 만든 GPL 저장소와 MIT 저장소로 동시에 존재할 수 있습니다. YOLOv9이 바로 그런 경우이며, 이 사실이 결정 전체를 바꿉니다.

이 가이드는 2026년 7월 기준으로 각 논문과 코드베이스 링크를 달아 저장소 단위로 전체 지형을 정리합니다. 아래의 모든 라이선스는 실제 저장소의 실제 LICENSE 파일과 대조해 확인했습니다. YOLO 라이선스에 관해 쓰인 글 중 상당수가(이 주제로 가장 높은 순위에 오른 가이드들을 포함해) 놀라울 만큼 틀렸기 때문입니다.

Ultralytics AGPL 질문에 대한 답만 필요하다면 [YOLO 상업적 사용, 무료일까?](/articles/yolo-commercial-license)에서 자세히 다룹니다. 이 글은 전체 지도입니다.

**시작하기 전에 밝혀 둘 두 가지.** 첫째, 저희는 아래 여러 프로젝트와 경쟁하는 MIT 라이선스 라이브러리인 LibreYOLO를 유지보수하며, 이 글의 마지막 섹션은 LibreYOLO에 관한 내용입니다. 그러니 저희 말을 그대로 믿기보다 직접 검증해야 할 이유이며, 그래서 이 글의 모든 라이선스 주장에는 1차 출처 링크를 달았습니다. 둘째, 이 글은 위 날짜 기준으로 공개된 라이선스 텍스트와 공개 발언에 관한 일반적인 정보입니다. 법률 자문이 아니며, 해당 관할 지역 변호사의 조언을 대신하지 않습니다. 라이선스와 메인테이너의 입장은 바뀝니다. 출시 결정을 내리기 전에 해당 저장소의 현재 LICENSE 파일을 읽으십시오.

## 라이선스 표

"허용적 경로"란 애플리케이션을 오픈 소스화하지 않고 라이선스 비용도 내지 않은 채 비공개 소스 상업 제품에 사용할 수 있는, 해당 모델을 구현한 저장소를 뜻합니다.

| 모델 | 연도 | 원본 코드 | 허용적 경로 | 논문 |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (퍼블릭 도메인) | 원본 그대로 | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (퍼블릭 도메인) | 원본 그대로 | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (퍼블릭 도메인) | 원본. AGPL PyTorch 포팅판에 주의 | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (퍼블릭 도메인) | 원본, [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | 없음 | 논문 없음 |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, 같은 연구실) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0, 출시 당시에는 GPL-3.0, 아래 참고) | 없음(Keras 포팅판은 중단됨, 아래 참고) | 논문 없음 |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, 같은 연구실) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | 없음 | 논문 없음 |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | 원본 그대로 | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | PaddleDetection의 원본(PaddleYOLO 아님) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (코드는 Apache-2.0, 가중치는 비상업용) | 코드는 가능, 공식 가중치는 불가 | 논문 없음 |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | 없음 | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, 2024년 이후 정체) | 없음 | 논문 없음 |

열 단위로 읽어 보면 한 가지가 분명해집니다. 버전 번호로는 라이선스를 전혀 알 수 없습니다. 알려 주는 것은 저장소입니다.

## 같은 모델, 두 개의 라이선스: YOLOv9 사례

YOLOv9은 "YOLOvN의 라이선스는 무엇인가?"가 잘못된 질문이라는 가장 깔끔한 증거이며, 그 경과는 정확히 짚어 볼 가치가 있습니다. YOLO 역사에서 커뮤니티의 압력이 실제로 결과를 바꾼 유일한 사례이기 때문입니다.

- **2024년 2월 18일:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)이 [논문](https://arxiv.org/abs/2402.13616)과 함께 공개됩니다.
- **2024년 2월 22일:** 한 사용자가 라이선스가 무엇인지 묻는 issue #10을 엽니다. Chien-Yao Wang(WongKinYiu)은 "I think it should be GPL3,"(GPL3이어야 할 것 같다)라고 답하고, 나흘 뒤 GPL-3.0 파일을 추가합니다.
- **2024년 2월 26일:** 두 번째 사용자가 [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82)를 엽니다. 이후 2주 반 동안 십여 명의 상업 사용자가 몰려듭니다.
- **2024년 3월 14일:** Wang이 그 스레드에 다음과 같이 씁니다. *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*(좋아요, MIT 재작성용 새 저장소를 만들겠습니다... 아키텍처와 손실 함수 구현은 대부분 제가 할 수 있을 것 같습니다. 데이터로더와 DDP 학습 쪽은 누군가 크게 도와주셔야 합니다.)

처음에 `yolov9mit`라는 이름이던 그 저장소가 오늘날의 **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)** 저장소입니다. MIT 라이선스이며, 저작권자는 "Kin-Yiu, Wong and Hao-Tang, Tsui,"이고, 스스로를 *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."*(YOLOv7, YOLOv9, YOLO-RD의 공식 구현체)라고 소개합니다. 재작성의 대부분은 스레드에 모인 자원자들이 아니라 Wang의 연구실 동료인 Hao-Tang Tsui가 맡았으며, 그가 전체 커밋의 약 90퍼센트를 작성했습니다.

따라서 YOLOv9은 어느 저장소를 클론하느냐에 따라 "출시할 수 없는" 모델이기도 하고 "출시할 수 있는" 모델이기도 합니다. 같은 아키텍처, 같은 연구실, 두 개의 라이선스입니다.

**이 저장소에 제품을 걸기 전에, 첫 페이지에는 나오지 않는 세 가지 주의 사항이 있습니다.**

1. **아직 성숙 단계입니다.** 2026년 2월에 열린 [미해결 이슈](https://github.com/MultimediaTechLab/YOLO/issues/231)는 제공되는 체크포인트가 논문의 COCO 수치를 재현하지 못하고, 논문에 적힌 것보다 파라미터가 눈에 띄게 많다고 보고합니다. 분할과 키포인트 학습은 [아직 구현되지 않았습니다](https://github.com/MultimediaTechLab/YOLO/issues/232). 2025년 12월 v1.0 태그 이후 main에 병합된 것이 없습니다.
2. **가중치에는 별도의 라이선스 명시가 없습니다.** 관례상 저장소의 MIT LICENSE가 릴리스 자산까지 포괄하며, 기술적 증거도 GPL 저장소에서 복사한 것이 아니라 이 프로젝트가 학습한 체크포인트임을 가리킵니다. 파일 크기와 파라미터 수가 GPL 저장소의 체크포인트와 다릅니다. 하지만 .pt 파일이 MIT라고 명시한 문서는 없으며, 커뮤니티의 [AGPL 오염 여부에 대한 공식 감사 요청](https://github.com/MultimediaTechLab/YOLO/issues/51)도 아직 열려 있습니다. 서류상의 공백입니다. 누구도 "깨끗함이 검증되었다"고 주장한 적이 없고, 누구도 문제를 제기한 적도 없습니다.
3. **다른 코드베이스입니다.** GPL 저장소의 스크립트를 그대로 대체할 수 있는 것이 아닙니다.

그렇다고 쓸 수 없다는 뜻은 아닙니다. 체크 표시 하나로 끝낼 대상이 아니라 평가가 필요한 프로젝트라는 뜻입니다.

## 1기: 퍼블릭 도메인 시대(YOLOv1부터 YOLOv4까지)

Joseph Redmon은 [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242)(YOLO9000 논문으로 발표), [YOLOv3](https://arxiv.org/abs/1804.02767) 모델을 자신의 Darknet 프레임워크 안에서 공개했으며, 그 [LICENSE 파일](https://github.com/pjreddie/darknet/blob/master/LICENSE)은 단 세 줄이라는 것으로 유명합니다.

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

이는 진정한 퍼블릭 도메인 헌정입니다. Alexey Bochkovskiy가 [Darknet 포크](https://github.com/AlexeyAB/darknet)에서 유지보수한 [YOLOv4](https://arxiv.org/abs/2004.10934)에도 *같은* 파일이 들어 있으며, 여러 라이선스 가이드가 주장하는 Unlicense가 아닙니다. GitHub의 자체 분류기는 이 파일을 판별하지 못하고 "Other"로 보고합니다. 컴플라이언스 도구가 이 필드를 읽는다면 알아 둘 만합니다. 텍스트는 더할 나위 없이 허용적인데도, 스캐너는 이를 깨끗한 허용적 라이선스가 아니라 식별되지 않은 라이선스로 표시합니다.

Darknet 자체도 완전히 사라진 것은 아닙니다. Redmon의 저장소는 2022년 이후 활동이 없지만, AlexeyAB의 README는 이제 권장되는, 활발히 유지보수되는 C/C++ 후속작으로 [hank-ai/darknet](https://github.com/hank-ai/darknet)을 안내합니다.

**함정은 포팅판입니다.** 2026년에 Darknet으로 학습하는 사람은 거의 없습니다. 대부분 PyTorch 재구현체를 쓰며, 이들은 각자 고유한 라이선스를 가집니다.

- 가장 인기 있는 YOLOv3 포팅판인 [ultralytics/yolov3](https://github.com/ultralytics/yolov3) 저장소는 **AGPL-3.0**입니다. 퍼블릭 도메인 아키텍처가 이 분야에서 흔히 쓰이는 라이선스 중 가장 엄격한 라이선스로 재배포되고 있습니다. 어떤 라이선스를 얻게 되는지는 전적으로 누구의 포팅판을 pip으로 설치했느냐에 달려 있습니다.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) 저장소는 **GPL-3.0**입니다.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)는 **Apache-2.0**이므로, v4의 허용적 경로는 PyTorch에서도 살아 있습니다.
- YOLOv4 공동 저자가 만든 [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)에는 **LICENSE 파일이 아예 없습니다**. 이는 카피레프트보다 나쁩니다. 라이선스가 없으면 어떤 권리도 부여되지 않으며, 기본값은 모든 권리 보유(all rights reserved)입니다.

논문의 버전 번호가 아니라 클론한 저장소의 라이선스를 확인하십시오.

## 2기: GPL-3.0(YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6)(Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv9](https://github.com/WongKinYiu/yolov9)은 GPL-3.0으로 배포되었습니다.

GPL-3.0은 카피레프트입니다. 수정한 버전을 배포하거나 GPL 코드를 자신의 코드와 결합해 하나의 프로그램으로 만든다면, 그 결합된 저작물의 완전한 대응 소스를 GPL-3.0으로 전달해야 합니다. 두 가지 경계가 중요합니다. **단순 집합**(mere aggregation)은 함께 배포될 뿐인 별개의 독립 프로그램이 자동으로 결합된 것은 아니라는 뜻입니다. 그리고 의무를 발생시키는 계기는 **배포**입니다: AGPL과 달리, 결합된 저작물을 전혀 배포하지 않는다면 일반 GPL은 공개 의무를 부과하지 않으며, 이 때문에 일부 회사는 GPL 모델을 철저히 자체 API 뒤에만 둡니다.

이 경로는 생각보다 좁고, 사람들이 좀처럼 대비하지 않는 세 가지 방식으로 무너집니다. 모델이 고객 기기나 온프레미스 설치 환경에 들어가는 순간 배포에 해당합니다. "내부 전용"은 사본이 자사 법인을 벗어나는 순간 더 이상 사실이 아닙니다. 한 회사 안에서 복사해 돌리는 것은 배포가 아니지만, 계약업체나 외주 개발팀, 별도 법인인 계열사에 빌드를 넘기는 것은 배포입니다. 그리고 이 방패는 해당 API에서 도달할 수 있는 *모든 것*이 GPL이거나 그보다 허용적일 때만 유효합니다. 제공되는 저작물 어딘가에 AGPL 구성 요소가 하나라도 있으면, GPL 파일 자체에 무엇이 적혀 있든 전체가 Section 13의 적용을 받기 때문입니다.

**YOLOv7과 YOLOv9에는 앞서 설명한 MIT 탈출구가 있습니다.** YOLOv6 모델에는 없으며, 그 가장 강력한 증거는 Baidu에서 나옵니다. PaddleDetection은 Apache-2.0이며, 메인테이너들은 YOLOv5, YOLOv6, YOLOv7, YOLOv8을 별도의 GPL-3.0 저장소인 [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)로 의도적으로 격리하면서, 이 모델들의 코드가 "will not be merged into PaddleDetection."(PaddleDetection에 병합되지 않는다)라고 밝혔습니다. Baidu만큼 큰 회사가 자사의 허용적 라이선스 프레임워크에서 YOLOv6 모델을 떼어 놓으려고 벽을 쌓는다면, 그 라이선스가 어떤 의미인지 알 수 있습니다.

## 3기: AGPL-3.0(YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics는 **2023년 4월 14일**에 AGPL-3.0으로 전환했습니다. 매우 널리 반복되는 것처럼(이 질문으로 검색 첫 페이지에 오르는 가이드들을 포함해) 2022년이 아닙니다. 그 이후의 모든 것([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11, [YOLO26](https://arxiv.org/abs/2606.03748))은 AGPL-3.0이며, 대안으로 유료 엔터프라이즈 라이선스(Enterprise License)가 있습니다. 이들의 [라이선스 페이지](https://ultralytics.com/license)는 Enterprise 등급이 "YOLO26, earlier YOLO versions, and any future YOLO models"를 포괄한다고 밝히므로, 이 정책은 의도적이며 앞으로도 적용됩니다. 가격은 공개되지 않습니다.

이 날짜는 확인할 수 있고, 확인할 가치도 있습니다. 널리 퍼진 이야기가 중요한 방식으로 틀렸기 때문입니다.

- `ultralytics/yolov5`의 `LICENSE` 파일은 역사상 정확히 세 번 수정되었습니다. 세 번째가 커밋 [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d) "Update LICENSE to AGPL-3.0 (#11359),"이며, 날짜는 **2023-04-14**이고, 101개 파일에서 `GNU GENERAL PUBLIC LICENSE`를 `GNU AFFERO GENERAL PUBLIC LICENSE`로 바꿉니다.
- 2022년의 마지막 YOLOv5 릴리스인 **v7.0(2022년 11월 22일)은 여전히 GPL-3.0으로 배포되었으며**, 그 이전의 v6.2, v6.1, v6.0도 마찬가지입니다. 2023년 4월 이전에 동기화를 멈춘 수천 개의 포크도 그렇습니다. 그중 아무거나 골라 보면 GitHub는 지금도 GPL-3.0으로 표시합니다.
- 같은 날 41분 뒤, `ultralytics/ultralytics`도 [같은 처리](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4)(PR #2031)를 받았습니다.

여기서 거의 아무도 모르는 사실이 나옵니다. **YOLOv8은 AGPL로 출시되지 않았습니다.** 2023년 1월 **GPL-3.0**으로 출시되었고 석 달 뒤 라이선스가 변경되었습니다. PyPI 기록은 명확합니다. `ultralytics` 8.0.0(2023년 1월 10일)부터 8.0.76(2023년 4월 13일)까지는 모두 `GPL-3.0`을 선언합니다. 2023년 4월 16일에 업로드된 버전 8.0.80이 `AGPL-3.0`을 선언한 첫 버전입니다.

이것은 빠져나갈 구멍이 아닙니다. 부여된 라이선스는 소급해 철회되지 않으므로, 그 옛 버전들은 공개 당시의 조건으로 계속 이용할 수 있습니다. 하지만 이들은 3년이나 지난, 유지보수되지 않고 패치도 없는 버전이며, GPL-3.0도 여전히 카피레프트이고(네트워크 조항을 배포 조항으로 바꾼 것일 뿐 카피레프트에서 벗어난 것이 아닙니다), 어차피 Ultralytics의 상업적 입장은 라이선스 텍스트보다 넓습니다. 유용한 교훈은 더 좁고 더 일반적입니다. **라이선스는 프로젝트의 이름이 아니라 받은 버전에 붙습니다.** 의존성 버전을 고정하고 사본을 받은 날 라이선스에 무엇이 적혀 있었는지 기록하십시오. 프로젝트는 내일이라도 라이선스를 바꿀 수 있고, 컴플라이언스 근거는 실제로 받은 버전이 어느 것이냐에 달려 있기 때문입니다.

AGPL-3.0은 GPL에 네트워크 조항인 Section 13을 더한 것입니다. *if you modify the Program*(프로그램을 수정하는 경우), 네트워크를 통해 수정된 버전과 상호작용하는 사용자에게 그 소스를 제공해야 합니다. 이로써 일반 GPL이 열어 둔 SaaS 경로가 막힙니다.

대부분의 글이 빠뜨리는 한정어에 주목하십시오. AGPL Section 0은 "modify"(수정)를 *in a fashion requiring copyright permission*(저작권 허락이 필요한 방식으로) 저작물을 복사하거나 개작하는 것으로 정의합니다. 수정하지 않은 학습 코드를 자체 데이터셋에 실행하는 것은 프로그램을 실행하는 것이지 그 소스를 개작하는 것이 아니며, 이는 수정하지 않은 GPL 컴파일러로 자신의 코드를 컴파일해도 컴파일러를 수정한 것이 아닌 것과 같습니다. 따라서 학습만으로 자동으로 의무가 발생하는 것은 아니며, 학습이 곧 의무 발생이라고 말하는 사람은 정의를 건너뛴 것입니다.

실제로 라이선스의 적용 범위에 들어가게 되는 것은, 패키지를 자신의 코드베이스에 결합하거나 포함시켜 둘이 하나의 프로그램으로 출시되거나 제공되는 경우입니다. FSF의 해석은 GPL 또는 AGPL 저작물을 다른 모듈과 링크하면 결합된 저작물이 된다는 것입니다. 법원이 이를 Python import까지 확장할지는 한 번도 다투어진 적이 없습니다. 또한 Ultralytics의 [상업 약관](https://ultralytics.com/license)은 내부적으로 YOLO를 사용하는 SaaS 플랫폼, API, 클라우드 시스템에 엔터프라이즈 라이선스를 요구하므로, 벤더의 입장은 라이선스 텍스트 자체보다 넓다는 점에도 유의하십시오. 제공하거나 출시하는 제품에 통합하는 것을 위험으로 취급하되, 학습만 따로 하는 것이 위험하다고 단정하지는 마십시오.

YOLO26 모델에 대해서는 곳곳에서 날짜가 뒤섞여 있으니 정리합니다. YOLO Vision에서 **2025년 9월 프리뷰로 공개**되었고, **실제 출시는 2026년 1월**(`ultralytics` 8.4.0 릴리스, 릴리스 노트에 "Ultralytics YOLO26 has arrived"라고 적혀 있음)이며, **논문은 2026년 6월**에 나왔습니다. YOLO26 모델은 메인 `ultralytics/ultralytics` 저장소에 있으며, `ultralytics/yolo26` 주소는 랜딩 페이지일 뿐입니다.

**사람들이 놀라는 부분:** 최근 2년간 나온 학술 YOLO들도 AGPL이며, 그 이유는 Ultralytics가 만들었기 때문이 아닙니다.

- [YOLOv10](https://arxiv.org/abs/2405.14458)은 칭화대학교에서 나왔습니다. README에는 *"The code base is built with ultralytics."*(코드베이스는 ultralytics로 만들어졌다)라고 적혀 있습니다.
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*(코드는 ultralytics를 기반으로 한다)
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*(코드는 Ultralytics를 기반으로 한다)
- YOLOv10 그룹이 만든 오픈 보캐뷸러리(open-vocabulary) "see anything" 모델인 [YOLOE](https://arxiv.org/abs/2503.07465)도 마찬가지로 AGPL-3.0이며 Ultralytics를 기반으로 만들어졌습니다.

카피레프트는 상속됩니다. AGPL 코드의 파생물은 AGPL로 전달해야 하므로, 이 포크들 중 하나가 출시되거나 제공되는 순간 라이선스도 함께 따라옵니다. (수정하더라도 철저히 혼자만 사용한다면 어떤 의무도 발생하지 않으며, 그래서 연구 목적의 사용에는 영향이 없습니다.) 연구는 독립적이지만 라이선스는 그렇지 않습니다. 2024년 이후 "다음 YOLO"를 Ultralytics 파생물로 발표하는 것이 기본 작업 방식이 되었고, 이는 이제 AGPL이 버전 번호를 따라 자동으로 전파된다는 뜻입니다.

YOLOv13이 Apache-2.0이라고 주장하는 페이지를 본다면 그것은 틀렸습니다. LICENSE 파일, GitHub API, 모델 카드 모두 AGPL-3.0이라고 말합니다.

### 더 이상 존재하지 않는 YOLOv8 허용적 경로

이 글의 초기 초안을 포함해 아직 온라인에 있는 많은 가이드는 AGPL YOLOv8에서 깔끔하게 벗어나는 방법으로 KerasCV의 Apache-2.0 YOLOv8을 제시합니다. **오늘날에는 이에 의존할 수 없습니다.** 공개된 기록은 다음과 같습니다.

- KerasCV 토론 [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032)에서 한 기여자는 "many parts are derived form ultralytics."(많은 부분이 ultralytics에서 파생되었다)라고 썼습니다. 메인테이너는 아무도 이 스레드에 답하지 않았으므로, 구현이 얼마나 독립적이었는지에 대한 질문은 공개적으로 어느 쪽으로도 결론 나지 않았습니다.
- KerasHub 이슈 [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760)은 2025년 7월 Keras 메인테이너가 "Hey all!! Because of license issues, we have decided to drop this."(여러분, 라이선스 문제로 이 작업은 중단하기로 했습니다.)라는 말과 함께 닫았습니다. 이후 두 사람이 KerasCV 버전의 기존 상업 사용자에게 이것이 무엇을 의미하는지 물었습니다. 둘 다 답을 받지 못했습니다.
- KerasCV는 현재 보관(archived) 상태이며, 유지보수되는 후속작인 KerasHub는 **YOLOv8을 제공하지 않습니다**.

저희는 이 두 가지가 연관되어 있는지 알지 못하며, 부적절한 일이 있었다고 주장하는 것도 아닙니다. 다만 제품을 쌓아 올릴 토대는 아닙니다.

## 허용적 라이선스 예외: YOLOX, PP-YOLOE, YOLO-NAS

모든 YOLO가 카피레프트 흐름에 올라탄 것은 아닙니다.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)**(Megvii, 2021)는 예외 조항 없이 코드와 가중치 모두 Apache-2.0입니다. 상업적 사용에 여전히 가장 깔끔한 고전 YOLO이지만, 저장소에는 2025년 중반 이후 커밋이 없고, [사전 학습된 가중치를 상업용 앱에 포함하는 것에 대한 질문](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865)은 2026년 3월부터 답변 없이 열려 있습니다. 라이선스 텍스트는 명확합니다. 다만 현재 그에 관한 질문에 답해 줄 사람이 없을 뿐입니다. [LibreYOLO로 YOLOX 실행하기](/articles/yolox-with-libreyolo)에 대해서도 글을 썼습니다.
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)**(Baidu)는 **PaddleDetection 안에서** Apache-2.0입니다. PaddleYOLO가 아니라 PaddleDetection에서 가져오십시오. PaddleYOLO에는 거의 동일한 config 디렉터리가 있지만 GPL-3.0입니다. 같은 모델, 같은 회사, 두 개의 저장소, 두 개의 라이선스입니다. YOLOv9 상황의 정반대입니다.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)**(Deci, 2023)는 사람들이 걸려 넘어지는 경우입니다. 코드는 Apache-2.0이지만 공식 가중치는 그렇지 않습니다. 이들의 [별도 라이선스](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)에는 *"may not use the Software for any commercial use, including in connection with any models used in a production environment."*(프로덕션 환경에서 사용되는 모든 모델과 관련된 사용을 포함해, 어떠한 상업적 용도로도 본 소프트웨어를 사용할 수 없다.)라고 적혀 있습니다. 2024년 NVIDIA가 Deci를 인수한 이후 저장소에서 새로운 기능 작업은 찾을 수 없었고, 원래의 가중치 URL은 더 이상 연결되지 않습니다(체크포인트는 현재 다른 호스트에 있습니다). 아키텍처를 처음부터 학습하면 Apache 조건만 남는다는 우회책이 가끔 제시되는데, 그럴듯하지만 Deci나 NVIDIA가 확인한 적은 없으며, "any components comprising the model"(모델을 구성하는 모든 구성 요소)까지 적용 범위로 삼는 라이선스 텍스트와는 어색하게 충돌합니다. YOLO-NAS에 대한 자세한 내용은 [여기](/articles/yolo-nas-with-libreyolo)를 참고하십시오.
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** 프로젝트는 오래된 가이드에서 Apache-2.0으로 소개되기도 합니다. 그렇지 않습니다. MMYOLO는 GPL-3.0이며(Apache인 쪽은 형제 프로젝트인 MMDetection입니다), 2024년 7월 이후 커밋이 없습니다.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)**(Tencent)는 AGPL도 아니고 허용적 라이선스도 아닌 **GPL-3.0**입니다. 굳이 언급하는 이유는, 오픈 보캐뷸러리 YOLO가 차용한 CLIP 계열 생태계 때문에 덩달아 허용적 라이선스일 것이라고 흔히 짐작되기 때문입니다.

이 사례들은 이 분야에서 가장 흔한 세 가지 라이선스 실수를 보여 줍니다. 가중치가 코드 라이선스를 따른다고 가정하는 것, 한 조직의 모든 것이 하나의 라이선스를 공유한다고 가정하는 것, 모델의 라이선스를 이름으로 짐작하는 것입니다.

## 라이선스는 아이디어가 아니라 코드를 다룬다(단, 특허는 주의)

위의 모든 허용적 경로의 법적 근거가 되는 원칙은 하나입니다. **저작권은 아이디어가 아니라 표현을 보호합니다.** 이는 구호가 아니라, 대서양 양쪽 모두에서 확립된 법 원칙입니다.

미국에서는 [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102)가 근거이며, 이 조항은 어떠한 "idea, procedure, process, system, method of operation, concept, principle, or discovery"(아이디어, 절차, 과정, 체계, 조작 방법, 개념, 원리 또는 발견)도 저작권 보호에서 제외하고, 그 뿌리는 *Baker v. Selden*(1879)까지 거슬러 올라갑니다. 저희에게 적용되는 법이자 아마 적지 않은 독자에게도 적용될 EU 법에서는, [소프트웨어 지침(2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) 제1조 제2항이 "ideas and principles which underlie any element of a computer program"(컴퓨터 프로그램의 어떤 요소의 기초가 되는 아이디어와 원리)는 보호되지 않는다고 규정하며, 유럽사법재판소는 *SAS Institute v World Programming*([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012)에서 프로그램의 기능, 프로그래밍 언어, 데이터 형식은 그 자체로 보호되는 표현이 아니라고 확인했습니다. 보호되는 것은 코드뿐입니다.

arXiv 논문에 설명된 아키텍처는 공개된 아이디어입니다. 누구나 구현할 수 있습니다. 할 수 없는 것은 남의 GPL/AGPL 소스 파일을 복사하거나 개작한 뒤 그 결과물의 라이선스를 바꾸는 것입니다. 소스가 아니라 논문을 보고 작업한 사람들이 작성한 독립적인 구현은 새로운 저작물이며, 라이선스는 그 작성자가 고릅니다.

대부분의 벤더가 건너뛰는 솔직한 주의 사항 두 가지:

- **"독립적"이라는 말이 사실이어야 합니다.** 이를 방어할 수 있는 형태는 클린룸 프로세스입니다. 코드를 작성하는 사람들이 카피레프트 소스가 아니라 논문과 명세를 보고 작업하는 방식입니다. GPL 저장소를 꼼꼼히 읽은 다음 "다시 쓰는" 것은 이와 같지 않으며, 누군가 들여다보게 된다면 그 차이가 중요해집니다.
- **이는 저작권 논리이지 특허 논리가 아닙니다.** 독립적 발명은 특허에 대한 방어 수단이 *되지 않습니다*. 클린룸 재구현은 저작권 문제를 해결할 뿐, 특허 문제는 그대로 남겨 둡니다. 오늘날 YOLO 아키텍처에 대해 널리 주장되는 특허는 없지만, "아직 아무도 소송을 걸지 않았다"는 것이 "소송을 걸 거리가 없다"는 뜻은 아닙니다.

## 가중치는 두 번째 라이선스

이 분야 전체에서 가장 비싼 실수는 코드를 감사하고 체크포인트를 잊는 것입니다. 라이선스가 붙는 층은 세 가지이며, 모두 다를 수 있습니다.

**1. 코드.** 위에서 다룬 모든 내용입니다.

**2. 가중치.** YOLO-NAS처럼 명시적인 경우도 있습니다. 주장되는 경우도 있습니다. Ultralytics의 [라이선스 페이지](https://ultralytics.com/license)는 AGPL-3.0이 "covers the training code and the models produced by that training code,"(학습 코드와 그 학습 코드로 만들어진 모델을 포괄한다)라고 밝히며, 이를 그들의 코드로 사용자가 직접 학습한 모델에까지 확장합니다. 자신들이 공개하는 것의 저작권자로서, 이들은 자체 체크포인트의 조건을 정할 수 있습니다. 사용자가 *직접* 만든 가중치가 법적으로 학습 코드의 2차적 저작물인지는 진정으로 열려 있는 질문입니다. 판례가 없고, 프로그램의 출력물은 프로그램의 저작권에 자동으로 포함되지 않는다는 FSF 자체의 일반 원칙과도 어긋납니다. 다만 정리되지 않았다는 것이 안전하다는 뜻은 아닙니다. 실무에서는 공개된 AGPL 가중치를 제약이 걸린 것으로 취급하십시오.

이 글의 조언을 따르는 사람이 조심해야 할 날카로운 지점이 있습니다. **GPL인 YOLOv7, YOLOv9 저장소는 가중치에 대해 아무것도 말하지 않습니다.** 이 체크포인트들은 별도의 허락 없이 GPL-3.0 저장소에 있는 릴리스 자산입니다. 그 .pt 파일을 MIT 재구현체에 불러오면, 코드는 깨끗하지만 상태를 아무도 명확히 하지 않은 체크포인트를 쓰게 됩니다. 라이선스 때문에 전환했다면, 허용적 라이선스 프로젝트가 직접 학습한 가중치를 쓰거나 직접 학습하십시오.

**3. 학습 데이터.** COCO의 *어노테이션*은 CC BY 4.0이므로 레이블은 문제없습니다. *이미지*는 COCO가 라이선스를 부여할 수 있는 것이 아닙니다. 개별 조건이 제각각인 Flickr 사진들이며, COCO 컨소시엄은 이 사진들을 소유하지 않는다고 명시합니다. 업계 대부분은 이를 낮은 위험으로 보고 넘어가지만, "COCO는 괜찮다"는 어노테이션에만 해당하는 말입니다. COCO를 벗어나면 금세 더 엄격해집니다. 항공, 의료, 주행 데이터셋 중 상당수가 비상업용이며, 모델 코드의 라이선스가 허용적이라고 해서 이들이 세탁되지는 않습니다.

매번 세 층을 모두 감사하십시오.

## MIT와 Apache-2.0에 관한 메모

저희는 MIT 라이선스 라이브러리를 배포하므로, MIT가 그냥 최고의 라이선스라고 말하면 편할 것입니다. 솔직한 비교는 그보다 흥미롭습니다.

MIT에도 의무가 없는 것은 아닙니다. 배포하는 사본에 저작권 고지와 라이선스 텍스트를 유지해야 합니다. 실제 조건이지만 부담이 적은 조건입니다. 그리고 Apache-2.0에는 MIT에 없는 것이 있습니다. 모든 기여자의 **명시적 특허 허락**과, 저작물에 대해 소송을 제기한 사람의 허락을 종료시키는 보복 조항입니다. MIT는 특허에 대해 아무 말도 하지 않습니다. 특허 노출이 가장 큰 걱정인 회사라면 Apache-2.0이 둘 중 *더 안전한* 쪽이라고 볼 수 있으며, 허용적 라이선스 탐지 생태계의 상당 부분(YOLOX, RT-DETR, D-FINE, DEIM)이 Apache-2.0인 것도 바로 그 이유입니다.

MIT가 주는 것은 단순함과 호환성입니다. 두 라이선스 모두 소스 공개를 강제하는 일은 절대 없으며, 이 글이 실제로 다루는 축은 바로 그것입니다. MIT냐 Apache냐가 중요한 결정이라고 말하는 사람이 있다면, 그 사람은 무언가를 팔고 있는 것입니다. 중요한 결정은 허용적 라이선스냐 카피레프트냐입니다.

## 그래서 실제로 쓸 수 있는 YOLO는

- **비공개 소스 상업 제품:** 연구실의 [MIT 저장소](https://github.com/MultimediaTechLab/YOLO)를 통한 YOLOv7 또는 YOLOv9, 원본 중에서는 YOLOX나 PP-YOLOE(PaddleDetection에서), 또는 이 모든 것을 직접 조립하고 감사하고 싶지 않다면 LibreYOLO 같은 유지보수되는 MIT 프레임워크입니다. 애플리케이션 전체를 공개하거나 엔터프라이즈 라이선스를 구매할 것이 아니라면 GPL이나 AGPL은 모두 피하십시오.
- **오픈 소스 프로젝트:** 라이선스가 호환되는 한 무엇이든 됩니다. 프로젝트가 AGPL이라면 Ultralytics 계열이 자연스럽게 맞으며, 최신 연구(YOLOv13, YOLO26, YOLOE)도 그쪽에 가장 먼저 나옵니다.
- **연구와 벤치마크:** 라이선스의 제약이 거의 없습니다. 비교 대상 논문이 사용한 것을 사용하십시오.
- **"나중에 처리하자"는 계획:** 통하지 않습니다. 제품을 다 만든 뒤 AGPL 의존성을 걷어 내려면 가중치를 포함해 모든 것을 다시 학습하고, 다시 검증하고, 다시 내보내야 합니다. 저장소를, 그리고 그에 따라 라이선스를 먼저 고르십시오.

라이선스가 아니라 프레임워크 자체를 비교한 내용은 [2026년 최고의 Ultralytics 대안](/articles/best-ultralytics-alternatives)을 참고하십시오.

## MIT 선택지: LibreYOLO가 제공하는 것

밝혀 둡니다. LibreYOLO는 저희 프로젝트이며, 그 라이선스가 이 글이 존재하는 이유입니다.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo)는 하나의 MIT 라이선스 코드베이스**로, 아래의 탐지기들을 단일 API 뒤에서 제공하며 학습과 ONNX, TensorRT, OpenVINO, NCNN 내보내기가 내장되어 있습니다. 카피레프트도, 네트워크 조항도, 엔터프라이즈 등급도 없습니다.

각 계열의 업스트림과 그 라이선스를 포함한 전체 객체 탐지 라인업은 다음과 같으며, 각 허용적 경로가 정확히 어디에서 오는지 확인할 수 있습니다.

| LibreYOLO 내 이름 | 모델 | 업스트림 | 업스트림 라이선스 |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Darknet 시대의 YOLO들, PyTorch 구현 | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | 퍼블릭 도메인 |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT(GPL 저장소가 아닌 연구실 자체의 재작성본) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, 그리고 NMS 없는 엔드투엔드 변형 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT(GPL 저장소가 아닌 연구실 자체의 재작성본) |
| `LibreYOLO9P2` | 작은 객체용 stride-4 헤드를 갖춘 YOLOv9 | LibreYOLO 자체 구현 | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, 탐지 및 자세 추정 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | 코드는 Apache-2.0. 제한이 걸린 업스트림 가중치는 재배포하지 **않습니다** |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR 및 v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, 탐지, 분할, 자세 추정, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | N/S/M/L은 Apache-2.0. XL/2XL은 Roboflow의 Platform Model License가 적용되므로 Apache 크기만 제공합니다 |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | 코드는 Apache-2.0. 큰 크기는 Meta의 DINOv3 백본을 사용하며, 이는 Meta 자체의 비OSI 라이선스를 따릅니다. 이 라이선스는 재배포를 허용하지만 **군사, 핵, 첩보, 무기 용도의 사용을 금지합니다**. 해당 크기는 Apache가 아니라 `other`로 게시됩니다. 이중 용도(dual-use) 제품을 출시하기 전에 반드시 읽으십시오 |
| `LibreRTMDet` | RTMDet([MMDetection 없이](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0(GPL-3.0인 mmyolo가 아닌 Apache 형제 프로젝트) |
| `LibrePICODET` | PP-PicoDet | 아키텍처는 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), 체크포인트는 [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) 재포팅판 | Apache-2.0(둘 다) |
| `LibreEC` | EdgeCrafter, 탐지, 자세 추정, 분할 | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | 마이크로컨트롤러급 하드웨어용 중심점 탐지기 | LibreYOLO 자체 구현 | MIT |
| `LibreGroundingDINO` | Grounding DINO, 오픈 보캐뷸러리(추론) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, 오픈 보캐뷸러리(추론) | Google, `transformers` 경유 | Apache-2.0 |

이 열이 의도적으로 솔직하게 밝히는 두 가지가 있습니다.

**GPL이나 AGPL 소스는 코드베이스에 전혀 복사되지 않았습니다.** 위의 모든 계열은 퍼블릭 도메인, MIT, Apache-2.0 업스트림에서 만들어졌으며, 이것이 프레임워크를 MIT로 유지하는 규칙입니다. 한 아키텍처에 허용적 라이선스 저장소와 카피레프트 저장소가 모두 있다면 허용적 라이선스 쪽을 기반으로 만듭니다. 저희의 YOLOv9과 YOLOv7은 GPL 원본이 아니라 연구실의 MIT 저장소에서, 저희의 RTMDet은 GPL 라이선스인 MMYOLO가 아니라 Apache 라이선스인 MMDetection에서 유래합니다. YOLO-World는 꾸준히 요청받는데도 목록에 없습니다. GPL-3.0이어서 허용적 라이선스로 가져올 곳이 없기 때문입니다.

**허용적 라이선스가 무제한을 뜻하지는 않으며, 감사에서 발견되게 두기보다 미리 말씀드리는 편이 낫다고 생각합니다.** DEIMv2의 큰 크기는 군사, 핵, 첩보, 무기 용도의 사용 금지를 포함한 Meta의 DINOv3 조건을 물려받으며, 이중 용도에 조금이라도 가까운 분야에서 일한다면 실제 제약이 됩니다. 몇몇 연구 프리뷰 체크포인트는 비상업용 데이터셋(DOTA, VisDrone)으로 학습되었으며, 무료인 것처럼 슬쩍 배포하는 대신 Hugging Face에서 `cc-by-nc`로 레이블을 붙였습니다. 그리고 다른 모든 곳의 가중치에 적용하는 신중함은 저희 가중치에도 똑같이 적용됩니다. 저희의 YOLO9 체크포인트는 MIT 연구실 저장소 자체의 체크포인트에서 변환한 것이므로 그 체크포인트에 해당하는 사실을 그대로 물려받으며, 앞서 언급했듯이 그것은 "주장되었을 뿐 공식적으로 감사되지 않은" 상태입니다. MIT 라이선스는 저희가 작성한 코드를 다룹니다. 모든 체크포인트는 학습에 쓰인 것과 출발점이 된 것의 조건을 따르며, 이는 가중치별로 게시되어 있습니다.

탐지 외에도 같은 API로 인스턴스 분할과 시맨틱 분할, 자세 추정, 회전 바운딩 박스, 분류(MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), 단안 깊이 추정, 이미지 복원, 시선 추정을 다룹니다.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

같은 YOLO 워크플로, 라이선스 숙제는 없습니다.

GitHub에서 Star를 눌러 주십시오: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*상표 및 제휴 관계: YOLO, YOLOv8, YOLO11, YOLO26, Ultralytics는 Ultralytics Inc.를 포함한 각 소유자의 상표입니다. LibreYOLO는 독립 프로젝트이며, Ultralytics 또는 이 글에 언급된 다른 어떤 조직과도 제휴 관계가 없고, 후원이나 보증을 받지 않습니다. 제품명과 저장소 이름은 논의되는 소프트웨어를 식별하고 비교하기 위한 목적으로만 사용됩니다.*

*이 글은 일반적인 정보이며 법률 자문이 아닙니다. 2026년 7월 11일 기준이며, 비교 대상 벤더 중 한 곳이 작성했습니다. 출시 전에 현재 라이선스를 확인하십시오.*
