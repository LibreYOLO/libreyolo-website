---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: dự đoán và xuất theo Apache-2.0'
description: >-
  Chạy DINO-DETR trong LibreYOLO để phát hiện đối tượng. Cài đặt, dự đoán, xác
  thực và xuất ba kích thước denoising-anchor, tất cả đều dùng giấy phép
  Apache-2.0.
lead: >-
  DINO-DETR, được IDEA Research công bố với tên DINO, kết hợp huấn luyện khử
  nhiễu tương phản với lựa chọn query hỗn hợp trên sparse attention của
  Deformable DETR. LibreYOLO cung cấp ba kích thước chỉ để inference phát hiện.
keywords:
  - DINO-DETR
  - DINO
  - detection transformer
  - denoising anchor box
  - lựa chọn query hỗn hợp
  - phát hiện đối tượng
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() trả về dict thuần túy, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Cài đặt

DINO-DETR không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở, sử dụng cùng lõi multi-scale deformable attention thuần PyTorch như họ Deformable DETR của LibreYOLO.

```bash
pip install libreyolo
```

Việc cài đặt `libreyolo[hub-kernels]` là tùy chọn. Khi có package `kernels`, LibreYOLO tìm nạp một kernel multi-scale deformable attention đã biên dịch từ Hugging Face Hub tại runtime và dùng nó thay cho lõi thuần PyTorch; `LIBREYOLO_HUB_KERNELS=0` sẽ tắt lại tính năng này.

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `max_det` lọc lựa chọn query; `iou` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì decoder là bộ dự đoán tập hợp không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

DINO-DETR chỉ dành cho inference trong LibreYOLO. Upstream huấn luyện bằng khử nhiễu tương phản và phép ghép Hungarian; công thức đó không được triển khai ở đây, vì vậy `train()` phát sinh `NotImplementedError`.

## Biến thể

Có ba checkpoint, tất cả ở cùng độ phân giải đầu vào. `r50` và `r50s5` dùng chung backbone ResNet-50 và khác nhau ở số lượng tỷ lệ feature map được đưa vào decoder, tương ứng là bốn và năm. `swinl` thay backbone bằng Swin-L và cũng lấy mẫu năm tỷ lệ.

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

<provenance-box>

Ba checkpoint chính thức đến từ thư mục phát hành Google Drive của tác giả, không phải model card trên Hugging Face. Repo upstream khai báo Apache-2.0 ở cấp repo nhưng không đính kèm tệp giấy phép hoặc metadata giấy phép cho chính các checkpoint, vì vậy cơ sở phân phối lại là khai báo cấp repo đó thay vì một quyền cấp riêng cho checkpoint. Mọi mirror LibreYOLO đều cung cấp nguyên văn nội dung giấy phép Apache-2.0 upstream cùng thông báo giải thích điều này.

</provenance-box>

## Trích dẫn

<citation-block />
