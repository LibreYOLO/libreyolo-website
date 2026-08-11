---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 trong LibreYOLO: phân loại zero-shot và tạo embedding'
description: >-
  Dùng SigLIP2 trong LibreYOLO để phân loại ảnh zero-shot và tạo embedding
  ảnh/văn bản với cách chấm điểm sigmoid đa nhãn. Không cần huấn luyện.
lead: >-
  SigLIP2 là mô hình hai tower chấm điểm ảnh theo các prompt văn bản bằng
  sigmoid độc lập cho từng lớp, thay vì softmax dùng chung trên một tập nhãn cố
  định. LibreYOLO hỗ trợ mô hình để phân loại zero-shot và tạo embedding ảnh/văn
  bản mà không cần bước huấn luyện.
keywords:
  - SigLIP2
  - SigLIP 2
  - phân loại zero-shot
  - embedding ảnh
  - embedding văn bản
  - open vocabulary
  - đa ngôn ngữ
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Khi không gọi set_classes(), dự đoán bằng CLI dùng 1.000 tên lớp

        # ImageNet mà mô hình tải theo mặc định.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Chấm điểm sigmoid đa nhãn
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # Xác suất độc lập theo lớp: nhiều lớp hoặc không lớp nào có thể cùng
        # đạt điểm cao. Softmax (mặc định) chuẩn hóa chúng thành phân phối
        # đơn nhãn, khớp với hành vi của LibreCLIP.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embedding ảnh và văn bản
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Cả hai đều được chuẩn hóa L2, nên tích vô hướng thông thường là độ
        tương đồng cosine.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data là thư mục gốc ImageFolder có phần chia train/; tên các thư mục
        # trở thành prompt lớp zero-shot cho lượt chạy này.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Các nhãn set_classes() hiện tại và độ phân giải đầu vào được nhúng
        # vào đồ thị. Hãy xuất lại sau khi thay đổi một trong hai. multi_label
        # phải là False (mặc định) khi xuất.
    - label: CLI
      language: bash
      code: |
        # Không gọi set_classes() ở đây, nên thao tác này nhúng 1.000 lớp
        # ImageNet mặc định mà mô hình tải.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Xuất embedding
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" chỉ truy vết image tower; không cần lớp.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Cài đặt

SigLIP2 cần extra riêng, extra này cài thêm gói SentencePiece mà tokenizer đa ngôn ngữ sử dụng.

```bash
pip install "libreyolo[siglip2]"
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`set_classes()` là thao tác cơ bản biến mô hình thành bộ phân loại open-vocabulary: phương thức kết xuất từng nhãn vào mọi mẫu prompt, mã hóa và lấy trung bình kết quả, rồi lưu ma trận `[K, D]` thu được làm head phân loại trong bộ nhớ đệm để không phải tính lại theo từng ảnh. Gọi lại phương thức bất cứ lúc nào để đổi lớp. Nếu không gọi, LibreSigLIP2 tải với 1.000 tên lớp ImageNet-1k đã được đặt sẵn.

SigLIP chấm điểm từng lớp độc lập: `logit = scale * (image . text) + bias`. Theo mặc định, tập logit vẫn được đưa qua softmax để tạo phân phối đơn nhãn khớp với hành vi `top1`/`top5` của LibreCLIP. Truyền `multi_label=True` vào `set_classes()` (hoặc khi khởi tạo) sẽ chuyển sang các xác suất sigmoid độc lập, vì vậy nhiều lớp hoặc không lớp nào có thể đạt điểm cao trên cùng một ảnh. Tokenizer là mô hình SentencePiece đa ngôn ngữ (từ vựng Gemma), nên tên lớp ở ngôn ngữ khác tiếng Anh hoạt động theo cùng cách.

Với `task="embed"`, dự đoán trả về một vector ảnh đã chuẩn hóa L2 cho mỗi đầu vào thay vì xác suất lớp, còn `embed_text()` trả về các hàng văn bản đã chuẩn hóa trong cùng không gian vector, vì vậy tích vô hướng thông thường giữa chúng là độ tương đồng cosine. `iou` không tác động đến cả hai tác vụ; không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Đánh giá

`val()` đọc tên thư mục lớp trong phần chia `train/` của ImageFolder, gọi `set_classes()` với các tên đó, rồi đo độ chính xác zero-shot top-1 và top-5 theo cách chấm điểm softmax. Độ chính xác phụ thuộc vào cách tên lớp được diễn đạt thành prompt chứ không phụ thuộc vào cập nhật trọng số vì không có gì để huấn luyện. Việc đánh giá chỉ bao quát `task="classify"`; `task="embed"` không có bộ đánh giá tập dữ liệu.

<code-tabs name="val" />

## Xuất

<export-matrix />

Quá trình xuất nhúng trạng thái hiện tại của mô hình vào đồ thị cố định. Với `task="classify"`, các nhãn được `set_classes()` đặt gần nhất cùng độ phân giải tại thời điểm xuất được nhúng vào lớp tuyến tính cuối với scale và bias đã học, vì vậy đồ thị đã xuất là bộ phân loại ảnh `[B, K]` thông thường, không có text tower hay tokenizer; hãy xuất lại sau khi đổi lớp hoặc kích thước. Chưa triển khai xuất ở chế độ `multi_label=True`; trước tiên hãy đặt lại thành `False`. Xuất `task="embed"` chỉ truy vết image tower. Cả hai cần ONNX opset 14 trở lên, được trình xuất đặt theo mặc định.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này. Cả hai đều được chuyển đổi từ các checkpoint `siglip2-base-patch16-256` và `siglip2-so400m-patch14-384` theo Apache-2.0 của Google, không phải từ bất kỳ lượt huấn luyện COCO nào.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

