---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: phát hiện với từ vựng mở trong LibreYOLO'
description: >-
  Dùng InternVL3 trong LibreYOLO để phát hiện đối tượng với từ vựng mở. Dự đoán
  bằng bất kỳ nhãn văn bản nào; không hỗ trợ huấn luyện, xác thực và xuất.
lead: >-
  InternVL3 là mô hình ngôn ngữ lớn đa phương thức nguyên bản do OpenGVLab phát
  hành, cùng học thị giác và ngôn ngữ trong một giai đoạn huấn luyện sẵn.
  LibreYOLO bọc mô hình thành detector với từ vựng mở: mọi danh sách nhãn văn
  bản đều trở thành tập lớp đối tượng, không có head cố định và không cần tinh
  chỉnh.
keywords:
  - InternVL3
  - InternVL
  - vision-language model
  - phát hiện với từ vựng mở
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE


        model = LibreInternVL3(size="2b")


        # Từ vựng mở: dùng được mọi từ, không phải class head cố định. Thiết lập

        # được giữ qua mọi lệnh gọi predict()/track() sau đó cho đến khi đặt
        lại.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat thô
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Lối truy cập mức thấp bên dưới tiện ích phát hiện: câu hỏi tự do,
        # đếm hoặc mọi prompt mà wrapper box không bao quát.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Cài đặt

InternVL3 cần extra `vlm`, extra này kéo về `transformers` cho backbone dùng chat template.

```bash
pip install "libreyolo[vlm]"
```

## Dự đoán

`LibreInternVL3` là lớp Python, không phải checkpoint `.pt`: lớp không được tải qua factory `LibreYOLO()`, còn CLI `libreyolo` không phân giải nó. Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) cũng truy cập họ mô hình này qua bí danh, ví dụ `LibreVLM("internvl3-2b")`; lớp dùng bên dưới là lớp mà factory khởi tạo. Trọng số đến từ repo Hugging Face `-hf` riêng của OpenGVLab, không phải mirror LibreYOLO; lệnh gọi đầu tiên tải về và lưu chúng vào bộ nhớ đệm cục bộ, đồng thời ghi log một thông báo giấy phép duy nhất cho trọng số Qwen có kiểm soát truy cập trước khi tải.

<code-tabs name="predict" />

`result.boxes` chứa các detection đã parse như mọi họ mô hình khác. Độ tin cậy là placeholder: InternVL3 không phát ra điểm theo từng box, vì vậy mọi detection nhận cùng độ tin cậy hằng số, còn `conf=` chỉ loại các hàng dưới hằng số đó chứ không xếp hạng. `iou` loại các box gần trùng của cùng lớp đối tượng khi vượt độ chồng lấp cho trước, một tác dụng phụ khi giải mã greedy lặp lại đối tượng; đây không phải lượt NMS theo lớp đối tượng. Bỏ qua `set_classes()` thì từ vựng mặc định dùng tên COCO-80. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba kích thước 1b, 2b và 8b, tất cả là checkpoint `-hf` nguyên bản của OpenGVLab (backbone LLM Qwen, không phải kiến trúc hai nhánh mà bài báo InternVL gốc mô tả). Bộ công cụ benchmark của LibreYOLO chưa đo họ mô hình này, vì vậy không có số liệu độ chính xác đã công bố để so sánh; hãy chọn kích thước theo ngân sách tính toán của bạn.

LibreYOLO chỉ cung cấp họ mô hình này để dự đoán. `train()`, `val()` và `export()` đều phát sinh `NotImplementedError`: thay vào đó hãy tinh chỉnh ở upstream rồi tải kết quả, việc xác thực tập dữ liệu (dataset) bị bỏ qua vì độ tin cậy placeholder sẽ khiến COCO mAP gây hiểu nhầm, còn xuất nằm ngoài phạm vi của mô hình sinh không có state dict để trace.

## Giấy phép

<provenance-box>

Mã riêng của InternVL3 dùng MIT, có tính cho phép và có thể dùng trong sản phẩm thương mại lẫn mã nguồn đóng. Các checkpoint `-hf` mà họ mô hình này tải mang backbone LLM Qwen và được cấp phép riêng theo Qwen License của Alibaba Cloud: được tự do sử dụng, sửa đổi và phân phối lại với yêu cầu ghi công "Built with Qwen" hoặc "Improved using Qwen", đồng thời giới hạn sử dụng thương mại ở 100 triệu người dùng hoạt động hằng tháng; vượt mức này cần giấy phép riêng của Alibaba. LibreYOLO không host hoặc phân phối lại các trọng số này: `LibreInternVL3` tải trực tiếp kích thước tương ứng từ `OpenGVLab/InternVL3-<size>-hf` trên Hugging Face trong lần chạy đầu tiên và ghi log thông báo một lần về Qwen License trước khi tải.

</provenance-box>

## Trích dẫn

<citation-block />


