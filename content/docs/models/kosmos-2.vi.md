---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 trong LibreYOLO: phát hiện đối tượng có grounding'
description: >-
  Kosmos-2 trong LibreYOLO: cài đặt, thiết lập từ vựng mở và dự đoán box có
  grounding bằng mô hình dùng giấy phép MIT của Microsoft.
lead: >-
  Kosmos-2 là mô hình grounding của Microsoft: mô hình tạo chú thích cho ảnh rồi
  định vị từng cụm danh từ trong chú thích bằng box. LibreYOLO bọc mô hình thành
  detector với từ vựng mở: cung cấp danh sách lớp đối tượng khi dự đoán.
keywords:
  - Kosmos-2
  - vision-language model
  - grounding
  - phát hiện với từ vựng mở
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("kosmos-2")

        model.set_classes(["boat", "person"])


        # Bất kỳ nguồn nào thư viện chấp nhận: tệp, thư mục, URL, chỉ mục
        webcam,

        # luồng RTSP hoặc danh sách .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Cài đặt

Kosmos-2 thuộc cấp VLM-as-detector của LibreYOLO, một bề mặt sản phẩm tách biệt với các họ dựa trên checkpoint và có factory riêng. Mô hình cần extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. LibreYOLO tải trực tiếp repo `microsoft/kosmos-2-patch14-224` của Microsoft; khác với Florence-2, ở đây không cần bản đăng lại từ cộng đồng.

<code-tabs name="predict" />

Họ mô hình này tải qua factory `LibreVLM()`, không phải `LibreYOLO()`: các họ VLM không khai báo checkpoint loader, vì vậy cách định tuyến theo hậu tố tệp được mô tả trên các trang mô hình khác không áp dụng ở đây. `set_classes()` đặt từ vựng mà Kosmos-2 được yêu cầu tìm; thiết lập này được giữ lại qua mọi lần gọi `predict()`/`track()` sau đó cho đến khi bạn đặt lại. Kosmos-2 grounding các cụm danh từ thay vì khớp nhãn chính xác, vì vậy wrapper của LibreYOLO chấp nhận kết quả khớp một phần: lớp đối tượng tên `"boat"` cũng khớp cụm từ được tạo như "the boats". Mọi detection mang cùng độ tin cậy placeholder, vì vậy lọc `conf` là giữ tất cả hoặc bỏ tất cả thay vì xếp hạng, còn `iou` không có tác dụng vì wrapper xây dựng danh sách detection trực tiếp từ các thực thể đã grounding mà không có bước loại trùng. `chat()` phát sinh `NotImplementedError` vì Kosmos-2 được điều khiển bằng prompt `<grounding>` thay vì chat template. CLI của LibreYOLO không bao quát cấp này: không có dạng `libreyolo predict model=...` cho mô hình. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một kích thước `kosmos-2-patch14-224` ở 224 px, được tải dưới dạng `LibreVLM("kosmos-2")`. Đây là mô hình thuộc thế hệ năm 2023, và wrapper riêng của LibreYOLO ghi chú rằng grounding của mô hình thô hơn các detector mới trong cấp này.

LibreYOLO không huấn luyện, xác thực hoặc xuất Kosmos-2: `train()`, `val()` và `export()` đều phát sinh `NotImplementedError` cho mọi họ trong cấp này (xem cấp hỗ trợ bên trên). Hãy tinh chỉnh Kosmos-2 ở upstream rồi tải trọng số thu được nếu bạn cần đóng cố định từ vựng tùy chỉnh; kiểm tra trực quan đầu ra `predict()` thay vì lượt xác thực kiểu COCO vì mọi detection mang cùng độ tin cậy placeholder.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


