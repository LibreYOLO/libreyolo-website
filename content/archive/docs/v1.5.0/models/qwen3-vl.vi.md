---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL trong LibreYOLO: phát hiện với từ vựng mở'
description: >-
  Qwen3-VL trong LibreYOLO: cài đặt, thiết lập từ vựng mở, dự đoán hoặc chat
  bằng vision-language model Apache-2.0 của Alibaba.
lead: >-
  Qwen3-VL là vision-language model của Alibaba có grounding 2D nguyên bản.
  LibreYOLO bọc mô hình thành detector đối tượng với từ vựng mở và cung cấp trực
  tiếp tính năng chat tự do: cung cấp danh sách lớp đối tượng để phát hiện hoặc
  đặt câu hỏi.
keywords:
  - Qwen3-VL
  - vision-language model
  - phát hiện với từ vựng mở
  - grounding
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # Lối truy cập mức thấp bên dưới tiện ích phát hiện: mọi câu hỏi,

        # không chỉ truy vấn bounding box.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Cài đặt

Qwen3-VL thuộc cấp VLM-as-detector của LibreYOLO, một bề mặt sản phẩm tách biệt với các họ dựa trên checkpoint và có factory riêng. Mô hình cần extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. `LibreVLM()` được gọi không có đối số mặc định dùng Qwen3-VL-4B.

<code-tabs name="predict" />

Họ mô hình này tải qua factory `LibreVLM()`, không phải `LibreYOLO()`: các họ VLM không khai báo checkpoint loader, vì vậy cách định tuyến theo hậu tố tệp được mô tả trên các trang mô hình khác không áp dụng ở đây. `set_classes()` đặt từ vựng mà Qwen3-VL được yêu cầu tìm; thiết lập này được giữ lại qua mọi lần gọi `predict()`/`track()` sau đó cho đến khi bạn đặt lại. Mọi detection mang cùng độ tin cậy placeholder, vì vậy lọc `conf` là giữ tất cả hoặc bỏ tất cả thay vì xếp hạng; `iou` có tác dụng với họ mô hình này, loại box sau thuộc cùng lớp đối tượng khi chồng lấp với box đã giữ vượt ngưỡng vì nếu không trình sinh lặp lại có thể phát ra các box gần trùng cho một đối tượng. Khác với Florence-2 và Kosmos-2, Qwen3-VL cũng trả lời câu hỏi tự do qua `chat()`, cùng lối truy cập mức thấp được ghi lại trên factory `LibreVLM`. CLI của LibreYOLO không bao quát cấp này: không có dạng `libreyolo predict model=...` cho mô hình. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba kích thước Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct và Qwen3-VL-8B-Instruct, được tải dưới dạng `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` và `LibreVLM("qwen3-vl-8b")`. Cả ba khai báo đầu vào danh nghĩa 1024 px, nhưng cơ chế smart-resize riêng của processor Qwen quyết định canvas thực sự được truyền vào mạng, vì vậy số liệu này không phải độ phân giải vận hành cố định như ở các họ khác trên trang. LibreYOLO chưa công bố benchmark so sánh độ chính xác giữa ba kích thước.

LibreYOLO không huấn luyện, xác thực hoặc xuất Qwen3-VL: `train()`, `val()` và `export()` đều phát sinh `NotImplementedError` cho mọi họ trong cấp này (xem cấp hỗ trợ bên trên). Hãy tinh chỉnh Qwen3-VL ở upstream rồi tải trọng số thu được nếu bạn cần đóng cố định từ vựng tùy chỉnh; kiểm tra trực quan đầu ra `predict()` thay vì lượt xác thực kiểu COCO vì mọi detection mang cùng độ tin cậy placeholder.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


