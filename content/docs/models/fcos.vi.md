---
title: FCOS
families:
  - fcos
seo_title: 'FCOS trong LibreYOLO: dự đoán, xác thực và xuất'
description: >-
  Chạy FCOS trong LibreYOLO để phát hiện đối tượng không anchor. Cài đặt, dự
  đoán, xác thực và xuất bản port torchvision ResNet-50/FPN dùng BSD-3-Clause.
lead: >-
  FCOS phát hiện đối tượng theo từng pixel thay vì dựa vào tập anchor box định
  trước, dự đoán một box và điểm centerness tại mọi vị trí trên feature map.
  LibreYOLO port cách triển khai torchvision cho tác vụ phát hiện.
keywords:
  - FCOS
  - phát hiện không anchor
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

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Cài đặt

FCOS không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. Gọi mô hình mà không có đối số ngưỡng sẽ áp dụng các giá trị mặc định đã công bố của FCOS là `conf=0.2`, `iou=0.6` và `max_det=100`; hãy truyền bất kỳ giá trị nào trong ba giá trị để ghi đè. FCOS giữ bước NMS cuối trên các dự đoán theo từng pixel. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một kích thước: ResNet-50 với feature pyramid, biến thể duy nhất mà họ mô hình này nhận diện.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

FCOS xuất sang ONNX, TorchScript và OpenVINO. FCOS giữ tỷ lệ khung hình nguồn trước khi graph chạy, vì vậy LibreYOLO buộc `dynamic=True` cho các tuyến ONNX và OpenVINO bất kể giá trị được truyền để graph vẫn hợp lệ với shape đầu vào có padding. Tệp `.onnx` đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />
