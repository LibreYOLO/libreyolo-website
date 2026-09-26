---
title: API thị giác-ngôn ngữ
seo_title: 'API LibreVLM: bí danh, set_classes và chat'
description: >-
  Factory LibreVLM, mọi bí danh mô hình, từ vựng set_classes có tính duy trì,
  set_task, lối truy cập chat và lý do độ tin cậy chỉ là giá trị giữ chỗ.
lead: >-
  LibreVLM tải mô hình thị giác-ngôn ngữ sinh và vận hành như detector đối
  tượng. Danh sách lớp là prompt thay vì head cố định, và mô hình trả về cùng
  Results như mọi họ khác.
keywords:
  - LibreVLM
  - phát hiện bằng mô hình thị giác ngôn ngữ
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - trò chuyện libreyolo
last_verified: 1.5.0
verification: >-
  Bí danh được đọc từ libreyolo/models/vlm/__init__.py; repo, kích thước và danh
  sách tác vụ lấy từ các module họ trong libreyolo/models/vlm/ cùng
  libreyolo/models/sensenova/model.py; quy tắc lời gọi và lỗi phát sinh lấy từ
  libreyolo/models/vlm/base.py, tất cả ở v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Phát hiện với open vocabulary
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Đặt câu hỏi dạng tự do
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Cài đặt

Tầng này cần extra `vlm`.

<code-tabs name="install" />

## Factory

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` là bí danh, không phải đường dẫn. `**kwargs` được chuyển đến constructor
của họ, nhận `device`, `names` (từ vựng ban đầu, tương đương gọi `set_classes`
sau khi tải), `prompt` (ghi đè prompt phát hiện) và `max_new_tokens`. Bí danh
không xác định phát sinh `ValueError` liệt kê mọi bí danh.

<code-tabs name="usage" />

## Bí danh

| Họ | Bí danh | Kích thước | Trọng số |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot thượng nguồn đã ghim |

Bí danh mặc định là `qwen3-vl-4b`. Kích thước cho bí danh mặc định của từng họ
là mục được liệt kê đầu tiên: `qwen3-vl` phân giải thành `4b`, `lfm2-vl` thành
`450m`, `internvl3` thành `2b`, `smolvlm2` thành `2.2b`, `florence-2` thành `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` và `LibreMODUS` (cũng
được viết là `LibreModus`) được xuất ở cấp gói.

## Tác vụ

Hầu hết các họ chỉ phục vụ `detect`. Hai họ phục vụ nhiều tác vụ hơn:

| Họ | Tác vụ được hỗ trợ |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Vì tác vụ được điều khiển bằng prompt thay vì nhúng vào checkpoint, có thể chuyển
tác vụ trên mô hình đã tải:

```python
model.set_task(task: str) -> LibreVLMModel
```

Tác vụ được kiểm tra với danh sách hỗ trợ của họ, được duy trì qua các lời gọi
`predict()` và `track()` sau đó, còn mô hình được trả về để có thể nối chuỗi lời gọi.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Đặt open vocabulary. Mọi từ đều hoạt động vì mô hình được cung cấp prompt bằng
chúng thay vì bị giới hạn trong head cố định. Danh sách không được rỗng và các
mục phải duy nhất khi so sánh không phân biệt chữ hoa chữ thường. Truyền chuỗi
đơn phát sinh `TypeError` vì chuỗi sẽ bị duyệt thành các lớp một ký tự. Từ vựng
có tính duy trì: đặt một lần sau khi tải và giữ nguyên cho đến khi được đặt lại.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Sinh đa phương thức thô: ảnh và prompt vào, văn bản đã giải mã ra, nguyên văn.
Đây là lối truy cập linh hoạt bên dưới tiện ích phát hiện, dành cho câu hỏi dạng
tự do, đếm hoặc định dạng đầu ra mà wrapper phát hiện không bao quát.
`max_new_tokens` quay về `MAX_NEW_TOKENS` của họ, bằng 1024 trên lớp cơ sở. Quá
trình giải mã dùng greedy với mức phạt lặp nhẹ.

## Độ tin cậy

Đầu ra sinh không có độ tin cậy theo box đã hiệu chuẩn. Phiên bản này gán một giá
trị giữ chỗ hằng số để `predict`, vẽ và `track` hoạt động, khiến việc lọc bằng
`conf=` và mAP chỉ mang tính mềm thay vì có ý nghĩa. Đây cũng là lý do `val()`
phát sinh lỗi: COCO mAP trên điểm giữ chỗ sẽ gây hiểu lầm.

## Dự đoán và theo dõi

Bề mặt predict tiêu chuẩn được áp dụng và `track()` hoạt động, nên detector VLM
được đưa vào cùng pipeline như mọi họ khác. Hai chính sách cấp lớp khác detector
tích chập: tăng cường dữ liệu khi kiểm thử bị tắt vì tăng cường multi-scale không
có ý nghĩa với bộ sinh độ phân giải cố định; dự đoán theo batch cũng tắt vì quá
trình sinh là tự hồi quy và tiền xử lý trả về mã hóa văn bản-cùng-ảnh thay vì
tensor ảnh có thể xếp chồng.

## Không được hỗ trợ

`train()`, `val()` và `export()` phát sinh `NotImplementedError`. Hãy tinh chỉnh
ở thượng nguồn rồi tải trọng số kết quả.

## Mã từ xa

Mọi họ được phân phối đều tải qua lớp mô hình native, nên LibreYOLO không mặc định
thực thi mã repo bên thứ ba. Họ thực sự cần mã đó phải bật rõ ràng và ghim một
snapshot revision; LocateAnything là họ duy nhất làm vậy, được ghim vào commit
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS là ngoại lệ rõ ràng đối với schema checkpoint: bí danh phân giải thành
thư mục các tệp thượng nguồn đã ghim thay vì `.pt` LibreYOLO, và LibreYOLO không
thêm metadata v1.0 hay tái công bố thư mục đó.

