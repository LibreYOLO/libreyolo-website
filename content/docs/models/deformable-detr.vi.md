---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: dự đoán và xuất, Apache-2.0'
description: >-
  Chạy Deformable DETR trong LibreYOLO để phát hiện đối tượng. Cài đặt, dự đoán,
  xác thực và xuất năm kích thước sparse-attention, tất cả đều dùng giấy phép
  Apache-2.0.
lead: >-
  Deformable DETR thay thế cross-attention dense của DETR bằng lấy mẫu thưa, đa
  tỷ lệ quanh mỗi điểm tham chiếu, nhờ đó việc huấn luyện transformer detector
  trở nên khả thi. LibreYOLO cung cấp năm kích thước chỉ để inference phát hiện.
keywords:
  - Deformable DETR
  - detection transformer
  - sparse attention
  - attention đa tỷ lệ
  - phát hiện đối tượng
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() trả về dict thuần túy, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Cài đặt

Deformable DETR không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở, sử dụng lõi multi-scale deformable attention thuần PyTorch.

```bash
pip install libreyolo
```

Việc cài đặt `libreyolo[hub-kernels]` là tùy chọn. Khi có package `kernels`, LibreYOLO tìm nạp một kernel multi-scale deformable attention đã biên dịch từ Hugging Face Hub tại runtime và dùng nó thay cho lõi thuần PyTorch; `LIBREYOLO_HUB_KERNELS=0` sẽ tắt lại tính năng này.

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `max_det` lọc lựa chọn query; `iou` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì decoder là bộ dự đoán tập hợp không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

Deformable DETR chỉ dành cho inference trong LibreYOLO. Upstream huấn luyện bằng phép ghép Hungarian và focal classification loss; công thức đó không được triển khai ở đây, vì vậy `train()` phát sinh `NotImplementedError`.

## Biến thể

Năm checkpoint bao phủ các cấu hình đã phát hành, tất cả ở cùng độ phân giải đầu vào. `r50ss` giới hạn attention ở một tỷ lệ đặc trưng duy nhất; `r50ssdc5` bổ sung giai đoạn backbone C5 giãn trên cấu hình đó. `r50` là cấu hình đa tỷ lệ mặc định, lấy mẫu trên bốn cấp feature map. `r50refine` bổ sung tinh chỉnh bounding box lặp lại qua các lớp decoder, còn `r50twostage` tạo region proposal ban đầu từ đầu ra encoder thay vì các query được học.

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

## Trích dẫn

<citation-block />
