---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet trong LibreYOLO: dự đoán, huấn luyện và xuất'
description: >-
  Chạy RTMDet trong LibreYOLO để phát hiện đối tượng và phân đoạn instance bằng
  RTMDet-Ins. Cài đặt, dự đoán, huấn luyện, đánh giá và xuất theo Apache-2.0.
lead: >-
  RTMDet là detector một giai đoạn dự đoán từ một prior dựa trên điểm cho mỗi vị
  trí lưới, không dùng anchor, qua một head có các phép tích chập dùng chung
  giữa các cấp feature. LibreYOLO hỗ trợ mô hình cho phát hiện và phân đoạn
  instance RTMDet-Ins.
keywords:
  - RTMDet
  - phát hiện đối tượng
  - phân đoạn instance
  - RTMDet-Ins
  - phát hiện không anchor
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Phân đoạn instance
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -seg trong tên tệp chọn mask head RTMDet-Ins,
        # nên không cần đối số task ở đây.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mask
        print(metrics["metrics/mAP50-95(B)"])   # box
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Cài đặt

RTMDet không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. Tên tệp có `-seg` tự phân giải thành tác vụ
RTMDet-Ins, và khi đó `result.masks` chứa các mask instance bên cạnh các box.
`conf` đặt ngưỡng độ tin cậy và `iou` đặt ngưỡng NMS. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Năm kích thước từ `t` đến `x` dùng chung một kiến trúc ở cùng độ phân giải đầu
vào. Họ này không có bảng benchmark tại đây: hãy so sánh kích thước theo dung
lượng tệp checkpoint trong bảng bên dưới.

## Huấn luyện

<code-tabs name="train" />

Tác vụ phát hiện được huấn luyện qua `train()`. Các thành phần QualityFocalLoss,
GIoU và DynamicSoftLabelAssigner được chuyển từ mmdetection thượng nguồn; forward
pass và bản xuất ONNX tương đương từng bit với nguồn đó, còn hậu xử lý khớp đầu
ra của mmdet trong phạm vi 0,001 mAP trên các tập con val2017.

Những phần chưa được kiểm tra theo docstring của chính `train()`: hội tụ khi
tinh chỉnh trên tập dữ liệu nhỏ, mức tương đương bài báo khi huấn luyện từ đầu,
hành vi multi-GPU, thông lượng Mosaic và MixUp có bộ nhớ đệm, việc chuyển pipeline
hai giai đoạn nghiêm ngặt của thượng nguồn và các ghi đè weight decay theo tham số
đặt decay bằng 0 cho tham số norm và bias.

RTMDet-Ins không có luồng huấn luyện. Gọi `train()` trên checkpoint `-seg` hoặc
với `task="segment"` sẽ phát sinh `NotImplementedError`; phân đoạn instance chỉ
hỗ trợ suy luận và đánh giá.

`train()` cũng chấp nhận đối số `pretrained`, nhưng giá trị không bao giờ được
đọc bên trong phương thức: quá trình huấn luyện luôn tiếp tục từ trọng số dùng
để khởi tạo mô hình, nên `pretrained=False` không khởi tạo lại mạng.

Nếu giữ nguyên các giá trị khác, trình huấn luyện chạy 300 epoch với AdamW ở
`lr0=0.004` và `weight_decay=0.05`, warmup 1 epoch theo lịch cosine, đồng thời
tắt Mosaic và MixUp trong 20 epoch cuối.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

Với checkpoint `-seg`, khóa `metrics/mAP50-95` thông thường chứa điểm mask, và
cùng lượt chạy cũng báo cáo box dưới `(B)` cùng mask dưới `(M)`, nên cả hai đều
có sẵn từ một lượt.

## Xuất

<export-matrix />

Tác vụ phát hiện xuất được sang hầu hết định dạng; phân đoạn instance hiện không
xuất được sang định dạng nào; ma trận trên phản ánh sự phân chia đó. Artifact
phát hiện đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

