---
title: Bắt đầu nhanh
seo_title: Bắt đầu nhanh với LibreYOLO
description: >-
  Chạy detector trên ảnh, tinh chỉnh trên dataset nhỏ và xuất sang TorchScript
  hoặc ONNX, tất cả trên CPU, với khoảng mười dòng Python.
lead: >-
  Lộ trình ngắn nhất qua LibreYOLO: dự đoán trên một ảnh, huấn luyện trên
  dataset nhỏ, rồi xuất kết quả. Mọi lệnh tại đây đều chạy trên CPU.
keywords:
  - hướng dẫn nhanh libreyolo
  - tutorial libreyolo
  - dự đoán libreyolo
  - huấn luyện libreyolo
  - xuất mô hình libreyolo
  - ví dụ yolo python
last_verified: 1.5.0
meta:
  - label: Cài đặt
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Phần cứng
    value: CPU là đủ cho mọi nội dung trên trang này
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tải checkpoint ở lần dùng đầu tiên, rồi cache trong weights/.
        model = LibreYOLO("LibreYOLO9t.pt")

        # Một ảnh trả về một object Results.
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video và stream
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True trả về lần lượt một Results cho mỗi frame thay vì tạo danh
        sách.

        # Thay đường dẫn bằng chỉ mục webcam, URL RTSP hoặc thư mục.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8 là dataset 8 ảnh đi kèm thư viện. Nó được tải
        # từ URL ở lần dùng đầu tiên nên không cần chạy script nào.
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Xác thực
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() trả về dict thuần túy, không phải object.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() trả về đường dẫn đã ghi.
        path = model.export(format="torchscript")
        print(path)

        # Factory định tuyến theo hậu tố file, nên artifact được nạp lại như
        # checkpoint và trả về cùng object Results.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Cài đặt

```bash
pip install libreyolo
```

Đó là tất cả những gì phần predict và train bên dưới cần. Xuất sang ONNX cần
thêm một gói bổ sung; xem [cài đặt](/docs/install) để biết danh sách đầy đủ.

## Predict

<code-tabs name="predict" />

`LibreYOLO()` là một factory. Nó đọc file, xác định trọng số thuộc họ nào và
trả về mô hình của họ đó, nên đổi sang detector khác chỉ cần thay một dòng.
Truyền `LibreYOLO9t.pt` không có thư mục sẽ tìm
`weights/LibreYOLO9t.pt` tương đối với thư mục làm việc và tải xuống đó nếu
file còn thiếu. Xem [checkpoint và trọng số](/docs/weights) để biết quy tắc tải
và cách làm việc offline.

`save=True` ghi bản sao đã chú thích dưới `runs/detect/`, vào thư mục `predict`
tăng dần theo mỗi lần chạy. `Results` trả về chứa `boxes`, còn `names` ánh xạ
chỉ mục lớp sang nhãn. Một đường dẫn ảnh trả về một `Results`; thư mục, danh
sách ảnh hoặc `stream=True` trả về danh sách hoặc generator của chúng.

## Huấn luyện

<code-tabs name="train" />

`data` là YAML dataset. `coco8.yaml` đi kèm thư viện nên snippet chạy nguyên
như đã dán; tên không được bundle sẽ được đọc như một đường dẫn. Dataset được
phân giải dưới `~/datasets` hoặc dưới `LIBREYOLO_DATASETS_DIR` khi biến này
được đặt.

Một lần chạy ghi vào `project/name`, mặc định là thư mục dưới `runs/train`, với
`weights/best.pt` và `weights/last.pt` bên trong. `train()` trả về dictionary
gồm `save_dir`, `best_checkpoint`, `last_checkpoint`, loss theo từng epoch và
metric xác thực theo từng epoch. Checkpoint đã huấn luyện được nạp qua
`LibreYOLO()` giống hệt checkpoint huấn luyện sẵn.

Không phải họ nào cũng huấn luyện được. Khi một họ chỉ cung cấp inference,
`train()` phát `NotImplementedError` và nói rõ điều đó. [Khái niệm cốt
lõi](/docs/concepts) giải thích ý nghĩa từng cấp hỗ trợ.

## Xuất

<code-tabs name="export" />

TorchScript không cần gì ngoài bản cài cơ sở. Mỗi đích khác có gói bổ sung
riêng, và phạm vi hỗ trợ tùy theo họ và tác vụ thay vì đồng nhất: xem [xuất và
triển khai](/docs/export).

Các đối số mọi định dạng đều nhận gồm `imgsz` (số nguyên hoặc cặp chiều cao và
chiều rộng), `batch` (mặc định 1), `half`, `int8` cùng YAML `data` để hiệu
chuẩn, `dynamic` (mặc định True), `simplify` (mặc định True), `opset`, `device`
và `output_path`. Khi bỏ `output_path`, file được ghi dưới `weights/` với tên
suy ra từ checkpoint.

## Bước tiếp theo

- [Khái niệm cốt lõi](/docs/concepts) về tác vụ, họ, kích thước và tên checkpoint.
- [Checkpoint và trọng số](/docs/weights) về tự động tải, dùng offline và an toàn khi nạp.
- [Nhập trọng số có sẵn](/docs/migrate) nếu bạn đã có checkpoint từ dự án upstream.
- [Tất cả mô hình](/docs/models) để chọn họ phù hợp với bài toán.
- [Huấn luyện](/docs/train), [Dự đoán](/docs/predict) và [Xuất](/docs/export) cho quy trình đầy đủ.
