---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: semantic segmentation trong LibreYOLO'
description: >-
  Dùng SegFormer trong LibreYOLO cho semantic segmentation ADE20K với các kích
  thước b0-b5. Cài đặt, dự đoán, huấn luyện và xuất; trọng số huấn luyện sẵn chỉ
  dùng phi thương mại.
lead: >-
  SegFormer là transformer semantic segmentation kết hợp encoder Mix Transformer
  (MiT) phân cấp với decode head all-MLP gọn nhẹ, tránh decoder nặng và
  positional encoding cố định mà transformer segmentation trước đây cần.
  LibreYOLO hỗ trợ mô hình cho một tác vụ, semantic segmentation, trên sáu kích
  thước.
keywords:
  - SegFormer LibreYOLO
  - semantic segmentation SegFormer
  - Mix Transformer MiT
  - transformer phân vùng ảnh
  - dataset ADE20K
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (tinh chỉnh)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Từ đầu
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Không có model_path: khởi tạo ngẫu nhiên, không tải gì. Đây là cách
        duy nhất

        # để có trọng số không mang điều khoản phi thương mại của checkpoint
        huấn luyện sẵn.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Dùng file đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố file, nên artifact đã xuất được nạp
        # như mọi checkpoint và trả về cùng object Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Cài đặt

SegFormer không cần gói bổ sung tùy chọn. Mọi thành phần được import đều có
trong bản cài cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và cache cục bộ.

<code-tabs name="predict" />

`result.semantic_mask` chứa bản đồ lớp dày đặc: `.data` là tensor `(H, W)` gồm
ID lớp ở kích thước ảnh gốc, còn `.classes` liệt kê ID lớp thực sự hiện diện.
`result.boxes` là `None` vì không có detection theo instance. `conf` và `iou`
được nhận để parity API nhưng không thay đổi đầu ra: mô hình trả về một lớp mỗi
pixel, không phải detection theo instance để lọc hoặc loại trùng. Xem [dự
đoán](/docs/predict) để biết source, streaming và xử lý kết quả.

## Biến thể

Sáu kích thước từ b0 đến b5, mở rộng chiều rộng và độ sâu encoder Mix
Transformer theo từng bước trong khi giữ cùng thiết kế decode head all-MLP.

<checkpoint-table />

## Huấn luyện

`train()` mặc định tinh chỉnh checkpoint đã công bố. Thay vào đó, không truyền
`model_path` cho `LibreSegformer(...)` để dựng encoder và head khởi tạo ngẫu
nhiên rồi huấn luyện từ đầu, đây là cách duy nhất tạo trọng số không mang hạn
chế phi thương mại của checkpoint huấn luyện sẵn (xem [Giấy
phép](#licensing)).

<code-tabs name="train" />

Khi giữ mặc định, trainer theo công thức ADE20K trong bài báo SegFormer: AdamW
ở learning rate cơ sở của backbone với decode head được huấn luyện ở mức gấp 10
lần, weight decay ở mọi nơi trừ LayerNorm và convolution vị trí Mix-FFN, cùng
lịch suy giảm tuyến tính có warmup. Khả năng hội tụ đầu cuối của các kích thước
lớn b3 đến b5 chưa được xác thực.

Xem [huấn luyện](/docs/train) để biết dataset, augmentation, multi-GPU và logger.

## Xác thực

`val()` trả về dictionary khóa `metrics/`: mIoU và độ chính xác pixel, đo trên
bất kỳ dataset nào theo định dạng đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố file, nên file
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`.
[Xuất](/docs/export) liệt kê đối số mọi định dạng đều nhận.

<code-tabs name="export" />

## Checkpoint

Mọi file trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Encoder và decode head của LibreSegformer là bản port PyTorch từ bản triển khai
SegFormer Apache-2.0 của Hugging Face Transformers, không phải từ
NVlabs/SegFormer: repo gốc của NVIDIA chưa bao giờ được đọc hoặc sao chép và
chỉ được ghi công ở đây để ghi nhận tác giả bài báo. Chỉ các checkpoint huấn
luyện sẵn bên trên mang hạn chế phi thương mại của NVIDIA; kiến trúc và mã riêng
của LibreYOLO luôn dùng MIT.

</provenance-box>

## Trích dẫn

<citation-block />
