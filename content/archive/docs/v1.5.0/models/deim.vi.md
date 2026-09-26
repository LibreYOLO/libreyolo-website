---
title: DEIM
families:
  - deim
seo_title: DEIM và DEIMv2 trong LibreYOLO
description: >-
  Dùng DEIM và DEIMv2 trong LibreYOLO để phát hiện đối tượng. Cài đặt, dự đoán,
  huấn luyện, xác thực và xuất, từ kích thước nửa triệu tham số trở lên.
lead: >-
  Một detection transformer được huấn luyện bằng phép ghép one-to-one dense, hội
  tụ trong số epoch ít hơn nhiều so với các công thức DETR làm nền tảng cho nó.
  LibreYOLO có hai phiên bản, được phân biệt bằng checkpoint bạn tải.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - detection transformer
  - DETR
  - phát hiện đối tượng
  - phát hiện thời gian thực
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Phiên bản là một phần của tên tệp và factory định tuyến theo

        # checkpoint, vì vậy cả hai đều tải theo cùng một cách.

        model = LibreYOLO("LibreDEIMv2pico.pt")


        # Bất kỳ nguồn nào thư viện chấp nhận: tệp, thư mục, URL, chỉ mục
        webcam,

        # luồng RTSP hoặc danh sách .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDEIMn.pt")


        # coco128.yaml tải mẫu 128 ảnh trong lần sử dụng đầu tiên. Hãy trỏ
        `data`

        # đến YAML dataset của bạn cho một lượt chạy thực tế.

        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Khi không đặt, epochs, batch, imgsz và lr0 lấy từ công thức đã phát
        hành

        # cho kích thước được tải.

        model = LibreYOLO("LibreDEIMv2pico.pt")

        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Cần extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() trả về dict thuần túy, không phải đối tượng
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: So sánh trên COCO
      language: bash
      code: |
        # coco-val-only.yaml tìm nạp 5000 ảnh val2017 và bỏ qua tập
        # huấn luyện. Tệp chứa script tải xuống nhúng, vì vậy cần
        # cấp quyền rõ ràng trừ khi dataset đã có ở máy.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Cần extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Cài đặt

Cả hai phiên bản đều không cần extra tùy chọn. Mọi thành phần mà chúng import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

Tinh chỉnh bằng adapter với `lora=True` là ngoại lệ và cần extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `max_det` lọc phép giải mã top-k trên các query và lớp đối tượng; không có bước NMS để điều chỉnh, còn `iou` được chấp nhận nhưng không được dùng. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Phiên bản 1 cung cấp năm kích thước, tất cả ở cùng kích thước đầu vào. Phiên bản 2 giữ lại năm tên đó và bổ sung ba kích thước nhỏ hơn là `atto`, `femto` và `pico`, trong đó hai kích thước đầu chạy nguyên bản ở kích thước đầu vào thấp hơn phần còn lại. Do đó năm mã kích thước tồn tại ở cả hai phiên bản và đặt tên cho các mô hình khác nhau; phiên bản được ghi trong tên tệp checkpoint.

<benchmark-table task="detect" />

<va-embed />

Phiên bản 1 giữ kiến trúc của D-FINE và thay mục tiêu phân loại bằng loss nhận biết khả năng ghép từ công thức one-to-one dense, vì vậy hai họ mô hình dùng chung gần như mọi key state dict và được phân biệt bằng metadata trong checkpoint. Phiên bản 2 giữ hợp đồng huấn luyện đó và kết hợp các backbone: HGNetv2 ở kích thước dưới `s`, và vision transformer DINOv3 với spatial tuning adapter ở `s` trở lên. Chính backbone đó thêm giấy phép thứ hai cho bốn checkpoint này, vì vậy hãy đọc [giấy phép](#licensing) trước khi phát hành một checkpoint.

## Huấn luyện

Quá trình huấn luyện bắt đầu từ checkpoint đã phát hành. `pretrained` không bao giờ đến được trainer: phiên bản 1 cảnh báo key không xác định rồi bỏ qua, phiên bản 2 loại bỏ key này. Không phiên bản nào cung cấp mô hình khởi tạo ngẫu nhiên.

<code-tabs name="train" />

Hãy tự truyền `lr0` trên phiên bản 1. Chữ ký `train()` Python mặc định là `4e-4`, mức từ công thức COCO đã phát hành, trong khi cấu hình huấn luyện của họ mô hình mang `1e-4` làm mặc định tinh chỉnh, và giá trị thấp hơn đó là giá trị CLI phân giải khi thiếu đối số. Cấu hình ghi lại phép đo làm căn cứ: ở kích thước batch thực sự dùng trong tinh chỉnh, trên các dataset nhỏ, learning rate COCO làm giảm rõ rệt hiệu quả transfer.

Phiên bản 2 tự phân giải các giá trị mặc định đó. Để trống `epochs`, `batch`, `imgsz` và `lr0` khiến mô hình đọc từng giá trị từ công thức đã phát hành cho kích thước được tải, nhờ đó các kích thước nhỏ huấn luyện ở độ phân giải đầu vào riêng mà không cần chỉ định, còn giá trị bạn truyền sẽ ghi đè công thức. `imgsz` là đối số bị ràng buộc: giá trị phải là bội số dương của 32, nếu không phiên bản 2 sẽ phát sinh lỗi trước khi bắt đầu lượt chạy.

Xem [huấn luyện](/docs/train) để biết về dataset, tăng cường dữ liệu (data augmentation), multi-GPU và logger.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

Các hàng trong bảng benchmark bên trên đến từ bộ công cụ benchmark của LibreYOLO; ghi chú dưới bảng đó ghi lại dataset đã tạo ra chúng và liên kết đến bản ghi các lượt chạy.

## Xuất

<export-matrix />

Ma trận bao quát cả hai phiên bản trên một trang: khi hai phiên bản khác nhau về một định dạng, ô hiển thị mức hỗ trợ yếu hơn, vì vậy nội dung ở đây không phóng đại khả năng của bất kỳ phiên bản nào bạn tải.

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box>
Bốn kích thước DEIMv2 từ S trở lên lấy backbone từ DINOv3, vì vậy repo trọng số của chúng mang cả Apache-2.0 và DINOv3 License của Meta, còn LibreYOLO cung cấp mã nguồn backbone DINOv3 theo cùng thỏa thuận đó. Phần còn lại của họ mô hình này, bao gồm mọi kích thước DEIMv2 dưới S, chỉ dùng Apache-2.0.
</provenance-box>

## Trích dẫn

<citation-block />

DEIMv2 là một bài báo riêng và có block trích dẫn riêng tại [github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation); hãy trích dẫn bài đó nếu bạn dùng checkpoint phiên bản 2.
