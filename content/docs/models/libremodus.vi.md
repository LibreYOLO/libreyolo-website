---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS trong LibreYOLO: phân tích ảnh any-to-any'
description: >-
  Dùng LibreMODUS trong LibreYOLO cho độ sâu, pháp tuyến, cạnh và phát hiện,
  đồng thời kết hợp chúng bằng any2any(). Chỉ dành cho suy luận; trọng số được
  tải từ EPFL-VILAB.
lead: >-
  LibreMODUS là bản tích hợp chỉ dành cho suy luận của checkpoint MODUS 14B-A7B,
  một mô hình any-to-any biến một đầu vào bắt nguồn từ ảnh thành đầu ra khác:
  RGB vào, độ sâu ra; độ sâu vào, pháp tuyến ra; bất kỳ dạng nào trong số đó
  cộng với một cụm từ sẽ cho ra các box. LibreYOLO hỗ trợ bốn tác vụ qua API
  predict tiêu chuẩn và một tập rộng hơn qua any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - ước lượng độ sâu
  - pháp tuyến bề mặt
  - phát hiện cạnh
  - phát hiện theo mô tả
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Khi không có từ vựng tùy chỉnh, detect giải mã các token nhãn COCO
        # của checkpoint thành các class id COCO-80 liên tiếp.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding theo cụm từ
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() chuyển phát hiện sang grounding theo cụm từ: từng cụm từ
        # chạy độc lập và trả về qua cùng giao diện Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # Một đến ba đầu vào bắt nguồn từ ảnh (rgb, depth, normal, canny/edge),
        # cùng văn bản phụ tùy chọn, được kết hợp hướng đến một đích.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # Grounding qua any2any() cần đầu vào văn bản chỉ rõ cụm từ.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Cài đặt

LibreMODUS cần extra riêng, extra này cài thêm `accelerate` để điều phối mô hình lớn theo yêu cầu của checkpoint.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO không phân phối lại hay tạo bản sao trọng số MODUS. Theo mặc định, khi tải mô hình `LibreMODUS`, các tệp cần thiết được tải trực tiếp từ `EPFL-VILAB/MODUS` tại một revision Hugging Face được ghim. Lần tải mới luôn cần tài khoản Hugging Face đã xác thực của chính người dùng, ngay cả khi cổng truy cập của nguồn thượng nguồn tạm thời mở. Hãy xem xét và chấp nhận các điều khoản thượng nguồn, sau đó xác thực:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Để tránh mọi yêu cầu mạng, hãy trỏ đến snapshot bạn đã có:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Thư mục đó phải chứa `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` và `merges.txt`. Xem phần Giấy phép bên dưới để biết các điều khoản của checkpoint cho phép những gì.

## Dự đoán

<code-tabs name="predict" />

API tác vụ tiêu chuẩn bao quát bốn tác vụ, mỗi tác vụ ánh xạ đến một đích MODUS: `depth` cho độ sâu tương đối (`result.depth_map`), `normal` cho pháp tuyến bề mặt (`result.normal_map`), `edge` cho cạnh kiểu Canny (`result.edges`) và `detect` cho các box COCO-80 (`result.boxes`), trừ khi `set_classes()` chuyển sang grounding theo cụm từ. `set_task()` chuyển đổi giữa chúng trên cùng mô hình đã tải. Công thức được phát hành dùng mười bước lấy mẫu luồng với hệ số hướng dẫn văn bản 4.0 và hướng dẫn ảnh 2.0; ghi đè chúng bằng `inference_steps=`, `inference_cfg=` và `inference_image_cfg=` khi khởi tạo.

`any2any()` mở ra bề mặt phân tích công khai rộng hơn: một đến ba đầu vào bắt nguồn từ ảnh (`rgb`, `depth`, `normal`, `canny`/`edge`), cộng với văn bản phụ tùy chọn, được kết hợp hướng đến một trong các đích gồm độ sâu, pháp tuyến, cạnh, cạnh bắt nguồn từ SAM, phát hiện COCO hoặc grounding theo cụm từ. Mọi đầu vào bắt nguồn từ ảnh phải mô tả cùng một khung hình đã căn chỉnh; LibreMODUS từ chối chiều rộng và chiều cao không khớp thay vì đổi kích thước từng đầu vào độc lập. `chain=(...)` tạo các đích trung gian và đưa chúng trở lại cùng ngữ cảnh, trong ngân sách huấn luyện ba điều kiện của checkpoint. `verify=N` (N >= 2) tạo N ứng viên và giữ ứng viên đạt điểm cao nhất trong phép kiểm tra tính nhất quán nội tại có ràng buộc, được cung cấp qua `result.verification_score`.

`dtype="bf16"` (mặc định) khớp với độ chính xác số của checkpoint được phát hành; `dtype="fp8"` lưu các trọng số tuyến tính đủ điều kiện trong thân decoder dưới dạng E4M3 với hệ số tỉ lệ theo từng kênh đầu ra, chuyển đổi một lần vào bộ nhớ đệm cục bộ tại `~/.cache/libreyolo/modus/fp8` và giải lượng tử hóa về dtype đầu vào cho từng phép nhân ma trận. Vì vậy, tùy chọn này đánh đổi bộ nhớ chứ không đánh đổi độ chính xác ở cấp activation.

`train()`, `val()` và `export()` đều phát sinh lỗi: LibreMODUS chỉ dành cho suy luận, không cung cấp đánh giá trên tập dữ liệu và không có luồng xuất ONNX, TensorRT hay TFLite. `predict()` theo batch và tăng cường dữ liệu khi kiểm thử cũng không được hỗ trợ; mỗi lời gọi xử lý một ảnh.

## Giấy phép

<provenance-box>

LibreYOLO không lưu trữ hay tạo bản sao checkpoint MODUS ở bất kỳ đâu, kể cả trên tổ chức Hugging Face của mình: khi tải, LibreYOLO luôn lấy revision được ghim trực tiếp từ EPFL-VILAB/MODUS hoặc đọc snapshot đã có trên đĩa tại `checkpoint_path`.

</provenance-box>

## Trích dẫn

<citation-block />

