---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision trong LibreYOLO: 7 tác vụ, một checkpoint'
description: >-
  Dùng SenseNova-Vision trong LibreYOLO để phát hiện, phân đoạn, toàn cảnh, tư
  thế, điểm, độ sâu và OCR từ một checkpoint sinh theo prompt.
lead: >-
  SenseNova-Vision là mô hình đa phương thức hợp nhất biểu diễn các tác vụ thị
  giác dưới dạng sinh theo prompt trên decoder dùng chung: box, điểm, keypoint
  và từ OCR xuất ra dưới dạng văn bản có tag, còn map độ sâu, mặt nạ và toàn
  cảnh xuất ra dưới dạng ảnh do decoder dựng. LibreYOLO tải mô hình qua LibreVLM
  và hỗ trợ bảy tác vụ từ một checkpoint 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - mô hình đa phương thức hợp nhất
  - Bagel
  - phát hiện theo prompt
  - nhận thức dense
  - phân đoạn theo tham chiếu
  - phân đoạn toàn cảnh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() chuyển tác vụ trên cùng mô hình đã tải.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Phân đoạn theo tham chiếu và toàn cảnh
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Phân đoạn dùng tham chiếu: cần cụm từ đích, không phải danh sách lớp
        đối tượng.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Khi không có từ vựng tùy chỉnh, tác vụ toàn cảnh dùng các danh mục

        # toàn cảnh COCO mà checkpoint đã được tinh chỉnh.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Điểm, tư thế và OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Khi chưa đặt từ vựng, tư thế dùng mặc định "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Cài đặt

SenseNova-Vision cần extra riêng, extra này kéo về `accelerate` để dispatch mô hình lớn mà checkpoint cần và trên nền tảng không phải macOS, kéo về `bitsandbytes` để tải 4 bit.

```bash
pip install "libreyolo[sensenova]"
```

Checkpoint được mirror trên Hugging Face trong tổ chức riêng của LibreYOLO và tự động tải trong lần sử dụng đầu tiên; checkpoint dùng CC BY-NC 4.0, chỉ dành cho mục đích phi thương mại, và loader in thông báo đó trước mỗi lần tải tự động. Xem phần Giấy phép bên dưới.

## Dự đoán

<code-tabs name="predict" />

Mỗi dự đoán là một lượt giải mã diffusion trên backbone Bagel-MoT dùng chung, vì vậy đây là mô hình năng lực thay vì mô hình thời gian thực: độ trễ theo ảnh sẽ cao hơn rõ rệt so với detector hoặc segmenter chuyên dụng. `dtype="auto"` (mặc định) tải bf16 trên GPU đủ bộ nhớ và dùng lượng tử hóa (quantization) NF4 4 bit ở nơi khác, cần `bitsandbytes`; truyền `dtype="bf16"` để buộc dùng độ chính xác đầy đủ trên GPU đủ lớn. `noise_seed=42` khi khởi tạo đặt seed cho diffusion sampler để đầu ra dense có thể tái tạo; truyền `noise_seed=None` để tắt seed.

Bảy tác vụ dùng chung một checkpoint đã tải: `set_task()` chuyển giữa chúng mà không tải lại. `set_classes()` đặt từ vựng đang hoạt động; phát hiện, điểm, tư thế và toàn cảnh nhận danh sách lớp đối tượng, còn phân đoạn dùng tham chiếu và cần chính xác cụm từ cần tách. Mỗi tác vụ trả về đối tượng `Results` tiêu chuẩn với payload khác nhau được điền: `boxes` cho detect, `points` cho point, `boxes` và `keypoints` cho pose, `ocr` cho OCR, `depth_map` cho depth, `masks` cho segment và `panoptic` (cùng `segments_info`) cho panoptic. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Checkpoint

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


