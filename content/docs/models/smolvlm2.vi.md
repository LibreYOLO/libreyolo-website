---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 trong LibreYOLO: phát hiện với từ vựng mở'
description: >-
  SmolVLM2 trong LibreYOLO: cài đặt, thiết lập từ vựng mở, dự đoán hoặc chat
  bằng vision-language model Apache-2.0 của Hugging Face.
lead: >-
  SmolVLM2 là vision-language model nhỏ của Hugging Face. LibreYOLO bọc mô hình
  thành detector đối tượng với từ vựng mở và cung cấp trực tiếp tính năng chat
  tự do: cung cấp danh sách lớp đối tượng để phát hiện hoặc đặt câu hỏi.
keywords:
  - SmolVLM2
  - vision-language model
  - phát hiện với từ vựng mở
  - mô hình đa phương thức nhỏ
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # Lối truy cập mức thấp bên dưới tiện ích phát hiện: mọi câu hỏi,
        # không chỉ truy vấn bounding box.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Cài đặt

SmolVLM2 thuộc cấp VLM-as-detector của LibreYOLO, một bề mặt sản phẩm tách biệt với các họ dựa trên checkpoint và có factory riêng. Mô hình cần extra `vlm`, extra này cũng kéo về `num2words`, một dependency của processor riêng của SmolVLM2.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Họ mô hình này tải qua factory `LibreVLM()`, không phải `LibreYOLO()`: các họ VLM không khai báo checkpoint loader, vì vậy cách định tuyến theo hậu tố tệp được mô tả trên các trang mô hình khác không áp dụng ở đây. `set_classes()` đặt từ vựng mà SmolVLM2 được yêu cầu tìm; thiết lập này được giữ lại qua mọi lần gọi `predict()`/`track()` sau đó cho đến khi bạn đặt lại. SmolVLM2 không cần ghi đè parser trong LibreYOLO: mô hình dùng cùng đầu ra chat-template kết hợp JSON như mặc định chung của cấp, vì vậy prompt phát hiện và định dạng box không dành riêng cho họ. Mọi detection mang cùng độ tin cậy placeholder, vì vậy lọc `conf` là giữ tất cả hoặc bỏ tất cả thay vì xếp hạng; `iou` có tác dụng, loại box sau thuộc cùng lớp đối tượng khi chồng lấp với box đã giữ vượt ngưỡng vì nếu không trình sinh lặp lại có thể phát ra các box gần trùng cho một đối tượng. SmolVLM2 cũng trả lời câu hỏi tự do qua `chat()`, cùng lối truy cập mức thấp được ghi lại trên factory `LibreVLM`. CLI của LibreYOLO không bao quát cấp này: không có dạng `libreyolo predict model=...` cho mô hình. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một kích thước trong registry là SmolVLM2-500M-Video-Instruct, được tải dưới dạng `LibreVLM("smolvlm2-500m")`. SmolVLM2 là detector yếu hơn các mô hình grounding chuyên dụng trong cấp này; wrapper riêng của LibreYOLO mô tả nó như ví dụ cho thấy họ mô hình mới không cần parse trường hợp đặc biệt để hoạt động ở đây, không phải tùy chọn từ vựng mở mạnh nhất.

LibreYOLO không huấn luyện, xác thực hoặc xuất SmolVLM2: `train()`, `val()` và `export()` đều phát sinh `NotImplementedError` cho mọi họ trong cấp này (xem cấp hỗ trợ bên trên). Hãy tinh chỉnh SmolVLM2 ở upstream rồi tải trọng số thu được nếu bạn cần đóng cố định từ vựng tùy chỉnh; kiểm tra trực quan đầu ra `predict()` thay vì lượt xác thực kiểu COCO vì mọi detection mang cùng độ tin cậy placeholder.

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


