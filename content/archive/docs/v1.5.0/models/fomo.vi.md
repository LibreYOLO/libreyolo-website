---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: định vị điểm, huấn luyện và xuất trong LibreYOLO'
description: >-
  Chạy FOMO (Faster Objects, More Objects) trong LibreYOLO: detector định vị
  điểm siêu nhỏ để đếm nhiều đối tượng nhỏ. Cài đặt, dự đoán, huấn luyện và
  xuất.
lead: >-
  FOMO là bộ định vị điểm dựa trên lưới: mỗi ô của lưới độ phân giải thấp được
  phân loại là hậu cảnh hoặc tâm đối tượng, không hồi quy bounding box.
  LibreYOLO hỗ trợ mô hình cho tác vụ điểm.
keywords:
  - FOMO
  - Faster Objects More Objects
  - định vị điểm
  - phát hiện tâm đối tượng
  - phát hiện vật thể nhỏ
  - AI biên
  - phát hiện trên MCU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Trọng số LibreFOMO không được tự động tải (xem phần Checkpoint bên
        dưới).

        # Hãy trỏ đến checkpoint bạn đã tải xuống cục bộ.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # Phải truyền imgsz: CLI mặc định dùng 640, còn checkpoint s

        # chỉ chấp nhận độ phân giải gốc 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Cài đặt

FOMO không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Không giống mọi họ khác trên trang này, trọng số LibreFOMO không được tự động
tải: `LibreYOLO("LibreFOMOs-point.pt")` tìm tệp đó trên đĩa và phát sinh
`ValueError` nêu rõ tên tệp thay vì lấy từ Hugging Face. Trước tiên, hãy tải
checkpoint từ [tổ chức LibreYOLO](https://huggingface.co/LibreYOLO) rồi tải bằng
đường dẫn cục bộ, hoặc tự huấn luyện (xem phần Huấn luyện bên dưới).

<code-tabs name="predict" />

Kết quả chứa payload `points` thay cho `boxes`: mỗi hàng là
`x, y, class, confidence`, có thể truy cập bằng `result.points.data` hoặc qua
các thuộc tính `.xy`, `.xyn`, `.cls` và `.conf`. Không có ngưỡng `iou` để đặt
vì không có box cần loại bỏ; `predict(..., nms_radius=1)` điều khiển khoảng cách
tối thiểu theo số ô lưới để hai phát hiện cùng được giữ lại, còn tên tệp phải có
hậu tố tác vụ `-point` của FOMO để trình tải nhận diện. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Ba kích thước `s`, `m` và `l` dùng các backbone kiểu MobileNetV2 rộng dần với
độ phân giải đầu vào cố định và lớn tương ứng, mỗi backbone đi kèm một head phân
loại 1x1 duy nhất. Họ này không có bảng benchmark tại đây; kích thước tệp
checkpoint trong bảng bên dưới là tín hiệu rõ nhất đã công bố cho từng kích thước.

## Huấn luyện

<code-tabs name="train" />

`imgsz` không phải lựa chọn tùy ý: giá trị mặc định là độ phân giải gốc của
checkpoint đã tải, và truyền giá trị khác sẽ phát sinh `ValueError` nêu rõ kích
thước mong đợi. Các kích thước đó là 96 cho `s`, 192 cho `m` và 224 cho `l`.
CLI mặc định đặt `imgsz` là 640, nên lệnh `libreyolo train` phải đặt rõ giá trị
khớp với checkpoint.

Nếu giữ nguyên các giá trị khác, trình huấn luyện chạy 40 epoch với batch 32,
Adam ở `lr0=3e-4`, không weight decay, và lớp tiền cảnh có trọng số gấp 100 lần
hậu cảnh trong loss cross-entropy theo ô vì gần như mọi ô lưới trong cảnh điển
hình đều là hậu cảnh. EMA và mixed precision đều tắt theo mặc định; không áp
dụng bất kỳ phép tăng cường hình học hoặc màu nào dùng ở nơi khác trong LibreYOLO:
mosaic, mixup, nhiễu HSV, lật, xoay, tịnh tiến và shear đều bằng 0.

Đây là luồng được dùng để huấn luyện các checkpoint LibreFOMO đã công bố từ đầu
trên COCO.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu và logger.

## Đánh giá

`val()` chuyển đến bộ đánh giá cấp lưới được xây dựng cho họ này. Bên cạnh các
khóa ghép điểm `metrics/precision`, `metrics/recall` và `metrics/mAP@` dùng
chung với các tác vụ điểm khác, phương thức quét các ngưỡng độ tin cậy và giá
trị `nms_radius`, rồi công bố tổ hợp có F1 tốt nhất trong `metrics/grid_F1`,
`metrics/grid_precision`, `metrics/grid_recall` và `metrics/grid_mean_distance`,
cùng ngưỡng và bán kính tạo ra kết quả đó trong `decode/threshold` và
`decode/nms_radius`.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này. Không tệp nào được tải tự động: hãy lấy
tệp bạn muốn từ trang Hugging Face được liên kết và truyền đường dẫn cục bộ vào
`LibreYOLO()`.

<checkpoint-table />

## Giấy phép

<provenance-box>

Không có repo mã thượng nguồn nào cho FOMO để liên kết: Edge Impulse mô tả kỹ
thuật qua bài blog và tài liệu sản phẩm nhưng chưa phát hành mã huấn luyện hoặc
suy luận FOMO. Kiến trúc và cách huấn luyện ở đây là bản triển khai riêng của
LibreYOLO dựa trên mô tả đã công bố, còn các checkpoint LibreFOMO đã công bố
được huấn luyện từ đầu trên COCO. Vì vậy, cả mã và các trọng số này đều theo MIT
và thuộc LibreYOLO. Tên FOMO cùng kỹ thuật mà nó mô tả vẫn thuộc Edge Impulse.

</provenance-box>

