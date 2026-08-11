---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 trong LibreYOLO: phát hiện với từ vựng mở'
description: >-
  Florence-2 trong LibreYOLO: cài đặt, thiết lập từ vựng mở và dự đoán box bằng
  mô hình thị giác dùng giấy phép MIT của Microsoft.
lead: >-
  Florence-2 là mô hình nền tảng thị giác của Microsoft, được điều khiển bằng
  task token thay vì chạy qua detection head cố định. LibreYOLO bọc mô hình
  thành detector với từ vựng mở: cung cấp danh sách lớp đối tượng khi dự đoán.
keywords:
  - Florence-2
  - vision-language model
  - phát hiện với từ vựng mở
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("florence-2-base")

        model.set_classes(["car", "person", "traffic light"])


        # Bất kỳ nguồn nào thư viện chấp nhận: tệp, thư mục, URL, chỉ mục
        webcam,

        # luồng RTSP hoặc danh sách .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Cài đặt

Florence-2 thuộc cấp VLM-as-detector của LibreYOLO, một bề mặt sản phẩm tách biệt với các họ dựa trên checkpoint và có factory riêng. Mô hình cần extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. LibreYOLO tải bản checkpoint do florence-community đăng lại thay vì repo `microsoft/Florence-2-*` gốc; xem phần Giấy phép để biết lý do.

<code-tabs name="predict" />

Họ mô hình này tải qua factory `LibreVLM()`, không phải `LibreYOLO()`: các họ VLM không khai báo checkpoint loader, vì vậy cách định tuyến theo hậu tố tệp được mô tả trên các trang mô hình khác không áp dụng ở đây. `set_classes()` đặt từ vựng mà Florence-2 được yêu cầu tìm trong ảnh; thiết lập này được giữ lại qua mọi lần gọi `predict()`/`track()` sau đó cho đến khi bạn đặt lại. Đối tượng `Results` trả về chứa `boxes` cùng shape như mọi họ khác, nhưng mọi detection mang cùng độ tin cậy placeholder, vì vậy lọc `conf` là giữ tất cả hoặc bỏ tất cả thay vì xếp hạng, còn `iou` không có tác dụng: wrapper Florence-2 xây dựng danh sách detection trực tiếp từ đầu ra task-token đã parse mà không có bước loại trùng. `chat()` phát sinh `NotImplementedError` ở đây vì Florence-2 được điều khiển bởi task token `<OPEN_VOCABULARY_DETECTION>` thay vì chat template. CLI của LibreYOLO không bao quát cấp này: không có dạng `libreyolo predict model=...` cho mô hình. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai kích thước Florence-2-base và Florence-2-large, cả hai ở 768 px, được tải dưới dạng `LibreVLM("florence-2-base")` hoặc `LibreVLM("florence-2-large")`. LibreYOLO chưa công bố benchmark so sánh độ chính xác giữa chúng.

LibreYOLO không huấn luyện, xác thực hoặc xuất Florence-2: `train()`, `val()` và `export()` đều phát sinh `NotImplementedError` cho mọi họ trong cấp này (xem cấp hỗ trợ bên trên). Hãy tinh chỉnh Florence-2 ở upstream rồi tải trọng số thu được nếu bạn cần đóng cố định từ vựng tùy chỉnh; kiểm tra trực quan đầu ra `predict()` thay vì lượt xác thực kiểu COCO vì mọi detection mang cùng độ tin cậy placeholder.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


