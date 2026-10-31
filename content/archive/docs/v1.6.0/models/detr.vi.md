---
title: DETR
families:
  - detr
seo_title: 'DETR: dự đoán và xuất theo Apache-2.0'
description: >-
  Chạy DETR, detection transformer nguyên bản, trong LibreYOLO. Cài đặt, dự
  đoán, xác thực và xuất bốn kích thước dựa trên ResNet, tất cả đều dùng giấy
  phép Apache-2.0.
lead: >-
  DETR là detection transformer nguyên bản, dự đoán một tập đối tượng cố định
  bằng transformer decoder ghép Hungarian thay vì anchor hay lưới dense.
  LibreYOLO cung cấp bốn kích thước chỉ để inference phát hiện.
keywords:
  - DETR
  - detection transformer
  - phát hiện đối tượng
  - phép ghép Hungarian
  - transformer decoder
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() trả về dict thuần túy, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Cài đặt

DETR không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `max_det` lọc lựa chọn query; `iou` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì decoder là bộ dự đoán tập hợp không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

DETR chỉ dành cho inference trong LibreYOLO. Upstream huấn luyện trong 500 epoch với phép ghép Hungarian; công thức đó không được triển khai ở đây, vì vậy `train()` phát sinh `NotImplementedError`.

## Biến thể

Bốn checkpoint kết hợp hai độ sâu backbone là ResNet-50 hoặc ResNet-101 với giai đoạn C5 giãn tùy chọn: các biến thể DC5 giữ giai đoạn backbone cuối ở độ phân giải đầy đủ thay vì tiếp tục downsample, vì vậy decoder đọc feature map mịn hơn từ cùng kích thước đầu vào. Cả bốn dùng chung 100 object query được học và transformer encoder-decoder sáu lớp, đồng thời đều chạy ở cùng độ phân giải đầu vào.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>
