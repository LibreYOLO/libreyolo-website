---
title: "LiteRT vs TensorFlow Lite: Đổi tên, không đổi định dạng"
description: "Google đổi tên TensorFlow Lite thành LiteRT vào tháng 9 năm 2024. Runtime vẫn như cũ, tệp .tflite vẫn như cũ, không có gì bị hỏng. Bài viết giải thích lý do đổi tên, những gì thực sự thay đổi và cách xuất mô hình LibreYOLO sang định dạng này."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite đã bị ngừng phát triển chưa?"
    a: "Tên gọi đang được thay thế, còn công nghệ thì không. Runtime tiếp tục tồn tại với tên LiteRT và cùng các API; những mô hình TFLite và đoạn mã hiện có vẫn hoạt động. Các phát triển mới, như API CompiledModel và khả năng tăng tốc NPU, được phát hành dưới tên LiteRT."
  - q: "Tôi có cần xuất lại mô hình để dùng LiteRT không?"
    a: "Không. Tệp .tflite được xuất cho TensorFlow Lite chính là mô hình LiteRT. Tệp không thay đổi, vì vậy bạn không cần xuất lại hay di chuyển mô hình."
---

**LiteRT là TensorFlow Lite với tên gọi mới. Đây không phải runtime mới hay định dạng tệp mới. Mô hình vẫn là tệp `.tflite`, và mọi tệp `.tflite` chạy được trên TensorFlow Lite đều chạy nguyên trạng trên LiteRT.**

Nếu bạn tìm kiếm "LiteRT vs TensorFlow Lite", "is TensorFlow Lite deprecated" hoặc "what is LiteRT", đoạn trên chính là câu trả lời. Phần còn lại của trang này giải thích ngắn gọn lý do đổi tên và cách xuất mô hình sang định dạng này bằng LibreYOLO.

## Vì sao Google đổi tên

TensorFlow Lite ra mắt năm 2017 như một cách chạy mô hình TensorFlow trên điện thoại và bo mạch nhúng. Qua nhiều năm, phạm vi của nó mở rộng vượt xa mục đích đó: Google xây dựng các quy trình chuyển đổi từ PyTorch, JAX và Keras, nên đến năm 2024, phần lớn mô hình chạy trên đó chưa từng dùng TensorFlow.

Lúc đó, tên gọi này gây hiểu nhầm rõ rệt. Người dùng PyTorch bỏ qua công cụ vì tưởng nó chỉ dành cho TensorFlow. Vì vậy, vào tháng 9 năm 2024, Google [đổi tên thành LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), viết tắt của "Lite Runtime". Toàn bộ câu chuyện chỉ có vậy. Cùng nhóm phát triển, cùng mã nguồn, tên gọi không gắn với framework cụ thể.

## Những gì thực sự thay đổi

Rất ít, và không có thay đổi nào làm hỏng thứ gì:

* Mã nguồn chuyển sang địa chỉ mới: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Tài liệu chuyển sang [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Gói Python để chạy mô hình hiện có tên `ai-edge-litert`. Gói `tflite-runtime` cũ đã lỗi thời; gói mới có cùng lớp `Interpreter`.
* Kể từ khi đổi tên, các tính năng mới được phát hành dưới tên LiteRT: API `CompiledModel` đơn giản hơn và khả năng tăng tốc GPU/NPU mà Google tuyên bố [sẵn sàng cho môi trường production vào tháng 1 năm 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

API Interpreter truyền thống cũng tiếp tục hoạt động như cũ. Có một phần mở rộng thực sự mới là `.litertlm`, nhưng đây chỉ là định dạng đóng gói cho LLM chạy trên thiết bị; với mô hình thị giác máy tính, định dạng này không liên quan đến bạn. Vì vậy, khi một tài liệu ghi "TFLite" còn tài liệu khác ghi "LiteRT", chúng đều nói về cùng một tệp.

## Xuất mô hình LibreYOLO sang LiteRT

LibreYOLO bổ sung quy trình xuất TFLite/LiteRT ở v1.3.0. Cần Python 3.12+ và cài thêm một gói:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # tự động tải về ở lần chạy đầu tiên
model.export(format="tflite")        # ghi ra tệp .tflite
```

Hoặc dùng CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Các đường dẫn đã được kiểm chứng hiện nay gồm phát hiện đối tượng YOLO9 và phát hiện đối tượng, phân đoạn, ước lượng tư thế RF-DETR; tất cả vẫn được đánh dấu là thử nghiệm. Ma trận hỗ trợ đầy đủ có trong [tài liệu](/docs/v1.3.0).

Để chạy tệp đã xuất, hãy dùng gói `ai-edge-litert` của Google trên thiết bị đích:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, ảnh đã tiền xử lý của bạn đặt ở đây
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Lưu ý đầu vào có bố cục NHWC (các kênh ở cuối): quá trình chuyển đổi từ ONNX hoán vị bố cục, đây là cách thông thường của định dạng này.

## Những gì chúng tôi đã thay đổi trong LibreYOLO

Không nhiều, vì cũng không cần thay đổi nhiều. Tài liệu hiện ghi định dạng là "TFLite (LiteRT)" để có thể tìm kiếm bằng cả hai tên, còn API vẫn dùng `format="tflite"` làm tên định dạng.

## FAQ

**TensorFlow Lite đã bị ngừng phát triển chưa?**
Tên gọi đang được thay thế, còn công nghệ thì không. Runtime tiếp tục tồn tại với tên LiteRT và cùng các API; những mô hình TFLite và đoạn mã hiện có vẫn hoạt động. Các phát triển mới, như API CompiledModel và khả năng tăng tốc NPU, được phát hành dưới tên LiteRT.

**Tôi có cần xuất lại mô hình để dùng LiteRT không?**
Không. Tệp `.tflite` được xuất cho TensorFlow Lite chính là mô hình LiteRT. Tệp không thay đổi, vì vậy bạn không cần xuất lại hay di chuyển mô hình.
