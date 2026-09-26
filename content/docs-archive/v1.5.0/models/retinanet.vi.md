---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet trong LibreYOLO: dự đoán, xác thực và xuất'
description: >-
  Chạy RetinaNet trong LibreYOLO để phát hiện đối tượng một giai đoạn bằng focal
  loss. Cài đặt, dự đoán, xác thực và xuất bản port torchvision dùng
  BSD-3-Clause.
lead: >-
  RetinaNet là detector một giai đoạn được huấn luyện bằng focal loss, giảm
  trọng số các mẫu âm dễ để lưới anchor dense không còn cần giai đoạn proposal
  riêng mà vẫn giữ độ chính xác. LibreYOLO port cách triển khai torchvision cho
  tác vụ phát hiện.
keywords:
  - RetinaNet
  - focal loss
  - phát hiện đối tượng
  - detector một giai đoạn
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Cài đặt

RetinaNet không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `iou` thiết lập ngưỡng độ tin cậy và NMS; RetinaNet giữ bước NMS upstream trên lưới anchor dense. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai kích thước, cả hai là ResNet-50 với feature pyramid: `r50` dùng head gốc, còn `r50v2` thay bằng GroupNorm head và block P6 rộng hơn nhận dữ liệu từ giai đoạn cuối của backbone thay vì đầu ra FPN.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

RetinaNet chỉ xuất sang ONNX với kích thước batch 1. RetinaNet đổi kích thước thành đầu vào biến đổi nhưng giữ tỷ lệ khung hình, vì vậy LibreYOLO buộc `dynamic=True` bất kể giá trị được truyền để graph vẫn hợp lệ với nguồn có shape khác nhau. Tệp `.onnx` đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>


