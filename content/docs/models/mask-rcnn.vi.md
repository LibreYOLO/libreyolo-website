---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN trong LibreYOLO: dự đoán, xác thực và xuất'
description: >-
  Chạy Mask R-CNN trong LibreYOLO để phát hiện đối tượng và phân đoạn thực thể.
  Cài đặt, dự đoán, xác thực và xuất bản port torchvision dùng BSD-3-Clause.
lead: >-
  Mask R-CNN thêm nhánh mặt nạ theo vùng vào Faster R-CNN, dự đoán mặt nạ phân
  đoạn cùng với mỗi box được phát hiện. LibreYOLO port cách triển khai
  torchvision cho phát hiện và phân đoạn thực thể.
keywords:
  - Mask R-CNN
  - phân đoạn thực thể
  - phát hiện đối tượng
  - Faster R-CNN
  - torchvision
  - detector hai giai đoạn
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Chỉ box
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" bỏ qua mask head và trả về box từ cùng checkpoint,
        # không có mặt nạ trong kết quả.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # mặt nạ
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Cài đặt

Mask R-CNN không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. Tải checkpoint mà không có đối số `task` sẽ trả về mặt nạ thực thể vì phân đoạn là tác vụ mặc định của họ này; khi đó `result.masks` chứa mặt nạ cùng các box. Truyền `task="detect"` tải cùng trọng số mà không có mask head và chỉ trả về box. `conf` và `iou` thiết lập ngưỡng độ tin cậy và NMS; Mask R-CNN giữ bước NMS upstream, khác với detector dựa trên query. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một backbone: ResNet-50 với feature pyramid, dùng builder Mask R-CNN v2 của torchvision. Checkpoint đã phát hành dùng giấy phép BSD-3-Clause và phục vụ cả hai tác vụ trong họ này, vì vậy không có kích thước để lựa chọn.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/`. Với tác vụ phân đoạn mặc định của checkpoint này, key thuần túy `metrics/mAP50-95` chứa điểm mặt nạ, còn cùng lượt chạy báo cáo box dưới hậu tố `(B)`, vì vậy cả hai có sẵn từ một lượt.

<code-tabs name="val" />

## Xuất

<export-matrix />

Mask R-CNN chỉ xuất sang ONNX với kích thước batch 1. Graph đã xuất giữ bước đổi kích thước và dán mặt nạ upstream bên trong, vì vậy LibreYOLO buộc `dynamic=True` bất kể giá trị được truyền để graph vẫn hợp lệ với nguồn không vuông. Tệp `.onnx` đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này. Checkpoint duy nhất bên dưới được liệt kê trong tác vụ phát hiện, nhưng cùng tệp cũng tải cho phân đoạn: không truyền đối số `task` và theo mặc định mô hình trả về mặt nạ.

<checkpoint-table />

## Giấy phép

<provenance-box>

Mask R-CNN được xây dựng dưới dạng lớp con của wrapper Faster R-CNN trong LibreYOLO: mô hình dùng chung nguồn torchvision và giấy phép BSD-3-Clause, đồng thời thêm mask predictor và mask RoI head từ cùng commit được port.

</provenance-box>


