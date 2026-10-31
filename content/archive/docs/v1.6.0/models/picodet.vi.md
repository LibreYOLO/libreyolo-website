---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet trong LibreYOLO: dự đoán, huấn luyện và xuất'
description: >-
  Chạy PicoDet trong LibreYOLO để phát hiện đối tượng trên thiết bị di động. Cài
  đặt, dự đoán, huấn luyện, đánh giá và xuất theo Apache-2.0.
lead: >-
  PicoDet là detector một giai đoạn được xây dựng cho CPU di động và thiết bị
  biên: backbone ESNet, neck CSP-PAN và head Generalized Focal Loss dùng chung.
  LibreYOLO hỗ trợ mô hình cho tác vụ phát hiện.
keywords:
  - PicoDet
  - PP-PicoDet
  - phát hiện đối tượng
  - phát hiện trên thiết bị di động
  - phát hiện trên thiết bị biên
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # Nên đặt imgsz: CLI mặc định dùng 640, còn checkpoint s có

        # độ phân giải gốc 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Cài đặt

PicoDet không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` đặt ngưỡng độ tin cậy và `iou` đặt
ngưỡng NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý
kết quả.

## Các biến thể

Ba kích thước, mỗi kích thước có độ phân giải đầu vào cố định riêng: `s` nhỏ
nhất và `l` lớn nhất. Độ phân giải tăng theo kích thước, vì vậy checkpoint lớn
hơn cũng tốn chi phí chạy trên mỗi ảnh hơn, ngoài việc chứa nhiều tham số hơn.

<benchmark-table task="detect" />

<va-embed />

## Huấn luyện

<code-tabs name="train" />

Các thành phần loss và assigner tuân theo công thức thượng nguồn: VFL, DFL,
GIoU và SimOTA, với trọng số chất lượng phân loại và đích VFL dùng IoU động.
Suy luận tương đương từng bit với thượng nguồn trên cùng checkpoint.

Những phần chưa được kiểm tra theo docstring của chính `train()`: hội tụ trên
toàn bộ tập dữ liệu, hành vi multi-GPU và mọi phép tăng cường dữ liệu ngoài lật
ngang. Checkpoint `s` ở độ phân giải gốc 320 cũng chưa vượt qua một cách ổn định
mức độ chính xác tối thiểu của LibreYOLO trên fixture 30 ảnh, hai lớp mà thư viện
dùng để kiểm thử các lượt tinh chỉnh nhỏ. Kích thước đó phù hợp hơn ở quy mô COCO đầy đủ.

`train()` cũng chấp nhận đối số `pretrained`, nhưng giá trị không bao giờ được
đọc bên trong phương thức: quá trình huấn luyện luôn tiếp tục từ trọng số dùng
để khởi tạo mô hình, nên `pretrained=False` không khởi tạo lại mạng. Nếu không
đặt `imgsz` trong Python, giá trị sẽ là độ phân giải gốc của checkpoint đã tải,
320 cho `s`, 416 cho `m` và 640 cho `l`. CLI luôn gửi `imgsz` với mặc định 640,
vì vậy hãy đặt giá trị khớp với checkpoint tại đó.

Nếu giữ nguyên các giá trị khác, trình huấn luyện chạy 300 epoch với SGD ở
`lr0=0.01`, momentum 0.9, weight decay 4e-5 và warmup 1 epoch theo lịch cosine.
Lật ngang là phép tăng cường dữ liệu duy nhất được áp dụng.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Bản chuyển đổi của LibreYOLO dựa theo Bo396543018/Picodet_Pytorch, một bản triển
khai lại PP-PicoDet gốc của PaddleDetection bằng PyTorch, đã loại bỏ mmcv và
khớp chính xác mọi activation để các checkpoint PaddlePaddle được chuyển qua
pipeline của Bo có thể tải mà không sai lệch số. Cả hai nguồn đều áp dụng cùng
điều khoản Apache-2.0 như các tác giả bài báo.

</provenance-box>

## Trích dẫn

<citation-block />

