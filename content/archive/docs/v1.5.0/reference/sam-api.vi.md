---
title: API promptable segmentation
seo_title: 'API LibreSAM: prompt, alias và signature'
description: >-
  Factory LibreSAM, alias kích thước, các loại prompt điểm, box và văn bản khái
  niệm, vòng đời encode một lần set_image và những gì cấp này không hỗ trợ.
lead: >-
  LibreSAM là factory cho promptable segmentation. Một forward pass cần prompt
  theo ảnh được cung cấp lúc gọi, nên cấp này có giao diện predict riêng thay vì
  định tuyến qua runner inference không prompt.
keywords:
  - LibreSAM
  - promptable segmentation
  - SAM point prompt
  - SAM box prompt
  - set_image
  - segment everything
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Alias factory, kích thước và repo được đọc từ libreyolo/models/sam/model.py,
  sam2.py, edgetam.py, sam3.py, libreyolo/models/mobilesam/model.py và
  libreyolo/models/picosam3/model.py. Contract prompt và giá trị mặc định được
  đọc từ libreyolo/models/sam/base.py. Ý định thiết kế từ
  docs/adr/0007-libresam-contract.md, tất cả ở v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompt điểm và box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Encode một lần, prompt nhiều lần'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Cài đặt

Cấp này cần gói bổ sung `sam`.

<code-tabs name="install" />

## Factory

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` là alias kích thước, không phải đường dẫn. `**kwargs` được chuyển đến
constructor của họ, nhận `device` và `multimask`. Alias không xác định phát
`ValueError` và thông báo liệt kê mọi alias đã biết.

<code-tabs name="usage" />

## Alias

| Họ | Alias | Kích thước | Trọng số |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, và các dạng ngắn `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Mặc định là `base`. SAM-1, SAM-2, EdgeTAM và MobileSAM chạy trên canvas danh
nghĩa 1024 pixel, SAM 3 trên 1008, PicoSAM3 trên 96.

Trọng số SAM 3 bị giới hạn truy cập. Chúng được tải từ `facebook/sam3` theo SAM
License tùy chỉnh của Meta, không phải MIT hoặc Apache-2.0 và không được
LibreYOLO phân phối lại. Hãy chấp nhận điều khoản trên trang repo và xác thực
với Hugging Face trước khi nạp; loader ghi log thông báo trước.

Các lớp họ cũng được export, nên có thể dựng trực tiếp `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` và `LibrePicoSAM3` bằng `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Đối số | Mặc định | Ý nghĩa |
|---|---|---|
| `source` | `None` | Ảnh cần segment; `None` tái sử dụng ảnh cache bởi `set_image()` |
| `points` | `None` | Prompt điểm theo tọa độ pixel |
| `bboxes` | `None` | Prompt box dạng `[x1, y1, x2, y2]` hoặc danh sách để tạo một mask mỗi box |
| `labels` | `None` | Nhãn điểm, `1` dương và `0` âm, có shape khớp `points`; tất cả dương khi bỏ qua |
| `masks` | `None` | Để dành; truyền giá trị sẽ phát `NotImplementedError` |
| `text` | `None` | Prompt khái niệm; chỉ SAM 3 |
| `conf` | `None` | Sàn mask-IoU dự đoán |
| `multimask` | `None` | Trả về mọi mask không chắc chắn theo prompt; mặc định theo thiết lập lúc dựng |
| `max_det` | `300` | Giới hạn mask trả về |
| `device` | `None` | Di chuyển mô hình cho lần gọi này và về sau, làm cache embedding mất hiệu lực |
| `color_format` | `"auto"` | Gợi ý định dạng màu cho mảng trong bộ nhớ |
| `points_per_side` | `None` | Mật độ lưới cho segment-everything; mặc định 32 |

Giá trị trả về là `Results` thông thường chứa `masks`, cùng `boxes` khít suy ra
từ mask, với lớp `0` tên `"object"`.

## Shape prompt

`points` nhận các dạng lồng `[x, y]` cho một object, `[[x, y], ...]` cho N
object và `[[[x, y], ...], ...]` cho điểm được nhóm theo object. Mảng Numpy
hoạt động ở mọi nơi danh sách hoạt động. Tọa độ là pixel thuần trên ảnh nguồn.

Bỏ mọi prompt không gian sẽ chạy segment-everything, trình tạo mask tự động
theo lưới có ngưỡng predicted-IoU và loại trùng box-IoU. Giá trị
`points_per_side` mặc định 32 chạy khoảng 1024 decoder pass, chậm trên CPU; hãy
giảm cho sử dụng tương tác. Generator bỏ lọc stability-score, multi-crop và
loại trùng mask-IoU, nên là xấp xỉ của pipeline có prompt chứ không khớp hoàn
toàn.

## Độ tin cậy

`conf` lọc theo mask-IoU dự đoán, là score chất lượng mask chứ không phải độ tin
cậy detection. `None` giữ mọi mask trong pipeline có prompt và áp dụng ngưỡng
lưới của họ trong segment-everything. `0.0` tắt lọc ở cả hai chế độ.

Trên pipeline văn bản SAM 3, `conf` là score detection Promptable Concept
Segmentation. `None` tại đó nghĩa là ngưỡng chuẩn 0.3, còn `0.0` giữ mọi ứng
viên.

## Prompt văn bản

`text=` chỉ dành cho SAM 3; mọi họ prompt không gian phát `NotImplementedError`
với nó. Văn bản loại trừ lẫn nhau với điểm và box. `names` trả về ánh xạ lớp `0`
đến khái niệm được yêu cầu. Lệnh gọi văn bản với `source=None` encode lại ảnh
cache vì tracker và concept encoder không dùng chung cache.

Keyword `exemplars=` được dành cho phần mở rộng image-exemplar trong tương lai
và chưa được triển khai.

## Vòng đời encode một lần

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` chạy image encoder nặng một lần và cache embedding, nên mọi
`predict()` sau với `source=None` đều rẻ. Cả hai phương thức trả về mô hình để
có thể nối lệnh gọi. Truyền `device=` cho `predict` di chuyển mô hình và làm
cache mất hiệu lực.

## PicoSAM3

PicoSAM3 chỉ nhận `bboxes=`. Prompt điểm, văn bản, mask, multimask và
segment-everything đều báo lỗi. Box được mở rộng 10 phần trăm và chạy qua mạng
ROI 96 pixel, còn PicoSAM3 là họ duy nhất trong cấp có thể xuất, chỉ sang ONNX.

## Không được hỗ trợ

`train()`, `val()` và `track()` phát `NotImplementedError` trên mọi họ trong
cấp. Mask promptable không có tập lớp cố định để chấm điểm nên mAP không có ý
nghĩa. `export()` báo lỗi cho SAM-1, SAM-2, SAM 3, EdgeTAM và MobileSAM.

Pipeline video và bộ nhớ cho SAM-2, SAM 3 cùng EdgeTAM nằm ngoài phạm vi phiên
bản này, tương tự image exemplar SAM 3 và prompt mask.
