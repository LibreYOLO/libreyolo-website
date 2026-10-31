---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN trong LibreYOLO: dự đoán, xác thực và xuất'
description: >-
  Chạy Faster R-CNN trong LibreYOLO để phát hiện đối tượng trên bốn backbone.
  Cài đặt, dự đoán, xác thực và xuất bản port torchvision dùng BSD-3-Clause.
lead: >-
  Faster R-CNN phát hiện đối tượng bằng mạng đề xuất vùng đưa vào bộ phân loại
  hai giai đoạn, kiến trúc đã biến region proposal thành một phần của cùng mạng
  được huấn luyện thay vì bước riêng. LibreYOLO port cách triển khai torchvision
  cho tác vụ phát hiện.
keywords:
  - Faster R-CNN
  - phát hiện đối tượng
  - region proposal network
  - detector hai giai đoạn
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Cài đặt

Faster R-CNN không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `iou` thiết lập các ngưỡng độ tin cậy và NMS; Faster R-CNN giữ bước NMS upstream, khác với detector dựa trên query. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có bốn kích thước, mỗi kích thước là một cấu hình torchvision khác nhau thay vì phiên bản được scale của cùng một cấu hình: `n` là MobileNetV3-Large ở đầu vào 320 px, `s` là cùng backbone ở 800 px, `m` là ResNet-50 với feature pyramid, còn `l` là bản sửa đổi v2 với region proposal head sâu hơn và box head bốn phép tích chập thay cho head của `m`. `n` và `s` đánh đổi độ chính xác để có backbone nhẹ hơn.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Faster R-CNN chỉ xuất sang ONNX với kích thước batch 1. Graph đã xuất giữ bước đổi kích thước upstream bên trong, vì vậy LibreYOLO buộc `dynamic=True` bất kể giá trị được truyền để graph vẫn hợp lệ với nguồn không vuông. Tệp `.onnx` đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
