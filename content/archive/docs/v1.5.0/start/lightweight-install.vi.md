---
title: Cài đặt gọn nhẹ
seo_title: Chạy inference LibreYOLO ONNX mà không cần PyTorch
description: >-
  Cài LibreYOLO với --no-deps và chạy detection ONNX chỉ bằng numpy, không có
  torch trên ổ đĩa. Kỹ thuật, giới hạn và danh sách package chính xác.
lead: >-
  Pipeline inference ONNX của LibreYOLO dùng numpy từ đầu đến cuối, bao gồm
  decode và NMS. Không phần nào cần PyTorch ở runtime, vì vậy bản cài bỏ qua
  phân giải dependency có thể chạy detection khi máy không có torch.
keywords:
  - inference không cần torch
  - libreyolo không cần pytorch
  - inference onnx không có torch
  - cài libreyolo gọn nhẹ
  - pip install no-deps
  - giảm dung lượng libreyolo
  - inference onnxruntime
last_verified: 1.5.0
meta:
  - label: Áp dụng cho
    value: 'Detection ONNX, bảy họ mô hình'
  - label: Điểm vào
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Mức hỗ trợ
    value: 'Nỗ lực tối đa, không phải bản phân phối riêng'
snippets:
  install:
    - label: Gọn nhẹ
      language: bash
      code: |
        # Cài package mà không lấy danh sách dependency, sau đó cung cấp bốn
        # package mà pipeline detection ONNX thực sự import.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Torch chỉ dùng CPU
      language: bash
      code: |
        # Hãy thử cách này trước. Nó giữ lại mọi tính năng và tránh wheel CUDA,
        # nơi chiếm phần lớn dung lượng ổ đĩa.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # xyxy ở đây là numpy ndarray, không phải torch tensor.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Vì sao cách này hoạt động

`pip install --no-deps libreyolo` cài package và bỏ qua hoàn toàn danh sách
dependency. Không có gì được phân giải thay bạn, và bạn chịu trách nhiệm cài
những gì mình thực sự dùng.

Điều đó chỉ hữu ích nếu code path bạn muốn thật sự không cần các dependency đã
bỏ qua, và detection ONNX đúng là như vậy. Quá trình decode, bao gồm non-maximum
suppression, dùng numpy. Các công thức tiền xử lý dùng numpy. PyTorch là
dependency cho huấn luyện và eager inference, còn pipeline này không bao giờ
gọi đến nó.

Trước bản phát hành này, import vẫn thất bại: import bất kỳ thứ gì dưới
`libreyolo.models` sẽ dựng mọi lớp mô hình để điền registry tự động nhận diện
checkpoint, trong khi các lớp đó là subclass của `torch.nn.Module`. Các công
thức tiền xử lý giờ nằm trong package riêng `libreyolo.preprocess`, và việc
import torch được hoãn đến khi có thành phần truy cập thuộc tính torch, nên
pipeline ONNX có thể import khi máy không có torch. Package này chứa một
preprocessor gốc numpy cho mỗi họ: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`,
`rfdetr`, `dfine`, `deim` và `deimv2`, nhiều hơn hai họ so với bảy họ được xác
thực đầu cuối bên dưới. Mỗi `libreyolo/models/<family>/utils.py` re-export từ
đó, nên các đường dẫn import hiện có vẫn hoạt động.

## Hãy thử wheel chỉ dùng CPU trước

Hầu hết người tìm cách này muốn tránh bản cài nhiều gigabyte, và dung lượng tập
trung ở một nơi: wheel `torch` mặc định kèm theo CUDA. Bản build chỉ dùng CPU
nhỏ hơn nhiều và không cần quy trình cài đặt đặc biệt.

<code-tabs name="install" />

Tùy chọn chỉ dùng CPU giữ lại mọi tính năng LibreYOLO: huấn luyện, xác thực,
mọi tác vụ, mọi họ và CLI. Chỉ chọn hướng gọn nhẹ khi bạn muốn máy hoàn toàn
không có torch, không chỉ muốn giảm dung lượng.

## Phạm vi của bản cài gọn nhẹ

| | |
|---|---|
| Tác vụ | Detection |
| Định dạng | ONNX |
| Điểm vào | `OnnxBackend` |
| Giao diện | Thư viện Python |

Bảy họ đã được xác thực trên pipeline này: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) và [DEIM](/docs/models/deim), tính cả các biến thể
của từng họ.

Đây là phạm vi đã xác thực, không phải ranh giới mà thư viện áp đặt. Các tác vụ
và họ khác đơn giản nằm ngoài phần đã kiểm tra: một số sẽ nạp torch khi được
gọi, và một vài trường hợp có thể vẫn hoạt động. Hãy coi mọi thứ ngoài danh
sách là chưa được kiểm thử, thay vì được hỗ trợ hoặc đã hỏng.

Trong phạm vi này, kết quả giống hệt bản cài thông thường chứ không chỉ gần
giống. Mỗi họ được xuất sang ONNX và chạy hai lần, một lần bình thường và một
lần chặn torch; box, score và class khớp hoàn toàn. Một parity test trong bộ
kiểm thử duy trì cam kết đó.

## Năm điểm thường gây nhầm lẫn

**Dùng `OnnxBackend`, không dùng các lớp mô hình.** `LibreYOLO9("model.onnx")`
vẫn cần torch vì bản thân `LibreYOLO9` là subclass của `nn.Module`. Đây là lỗi
dễ gặp nhất, bởi mọi trang khác trong tài liệu đều nạp mô hình qua lớp của nó
hoặc qua `LibreYOLO()`.

**Xuất ở máy khác.** Tạo file `.onnx` cần torch, nên máy gọn nhẹ không thể tạo
file này. Hãy xuất trên máy phát triển hoặc CI rồi chuyển artifact đến đích
gọn nhẹ.

**Kết quả chứa mảng numpy.** `result.boxes.xyxy` là một `ndarray` ở đây. Các
container nhận cả hai kiểu nên tên thuộc tính không đổi, nhưng mã gọi `.cpu()`
hoặc `.numpy()` trên kết quả sẽ thất bại.

**Một ảnh trả về một `Results`.** `predict()` trả về một `Results` cho một ảnh
và một danh sách cho nhiều ảnh. Lập chỉ mục kết quả đơn bằng `[0]` sẽ chọn
detection đầu tiên, không phải ảnh đầu tiên, nên âm thầm trả về kết quả chỉ có
một box thay vì báo lỗi.

**CLI không hoạt động.** `typer` và `click` không nằm trong bốn package, nên
lệnh `libreyolo` không khả dụng. Đây là bản cài thư viện.

## Predict

<code-tabs name="predict" />

Thay `onnxruntime` bằng `onnxruntime-gpu` để chạy trên CUDA. Bốn package là
những gì một lệnh `predict()` hoàn toàn không có torch thực sự import, được ghi
lại trong lúc gọi thay vì suy luận. `opencv-python-headless` thay cho
`opencv-python` được khai báo: cùng module, không có thư viện GUI và nhỏ hơn
trên ổ đĩa.

Trong số dependency được khai báo còn lại, `requests` chỉ cần khi nạp ảnh từ
URL, `pycocotools` và `scipy` phục vụ xác thực và đánh giá, còn `typer` và
`click` dành cho CLI.

## Danh sách này sẽ thay đổi theo thiết kế

Danh sách package trên đúng với bản phát hành nêu ở đầu trang. `--no-deps` đưa
bạn ra ngoài cơ chế phân giải dependency, nên không có gì kiểm tra thay bạn và
bản phát hành sau có thể import thành phần chưa được liệt kê.

Nếu gặp `ModuleNotFoundError`, bạn đã hiểu kỹ thuật này: hãy cài package còn
thiếu. Đó là mô hình bảo trì dự kiến, không phải bug report. Pipeline này được
hỗ trợ theo khả năng tốt nhất và không phải bản phân phối riêng, cũng là lý do
không có package gọn nhẹ thứ hai trên PyPI và không có kế hoạch tạo một package
như vậy.

Để xác nhận môi trường thật sự không có torch thay vì âm thầm dùng một bản đã
cài, hãy assert điều đó:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Kiểm tra này nên được giữ trong CI cho image gọn nhẹ. Nếu không, môi trường vô
tình có sẵn torch sẽ vượt qua mọi test nhưng không cho bạn biết điều gì.
