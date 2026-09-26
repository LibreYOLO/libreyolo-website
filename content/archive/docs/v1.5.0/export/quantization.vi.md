---
title: Lượng tử hóa
seo_title: Lượng tử hóa một mô hình LibreYOLO trong PyTorch
description: >-
  API lượng tử hóa (quantization) trong PyTorch của LibreYOLO: chín recipe, dữ
  liệu hiệu chuẩn tách riêng khỏi dữ liệu huấn luyện, QAT và QAD, cùng hai
  artifact triển khai.
lead: >-
  Quantization trong LibreYOLO chạy hoàn toàn trong PyTorch: model.quantize()
  thay các module Conv2d và Linear của mô hình bằng những module đã lượng tử hóa
  tương đương rồi hiệu chuẩn chúng. Kết quả vẫn giữ nguyên hợp đồng predict,
  val, train và save thông thường, nên một mô hình đã lượng tử hóa được chấm
  bằng đúng các validator dùng cho mô hình float.
keywords:
  - lượng tử hóa libreyolo
  - lượng tử hóa int8 pytorch
  - quantization aware training
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dữ liệu hiệu chuẩn quantization
  - xuất onnx qdq int8
last_verified: 1.5.0
meta:
  - label: Lệnh gọi
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Lệnh
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Phụ thuộc thêm
    value: Không có. Quantization chạy trong PyTorch.
  - label: Họ mô hình
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Các recipe
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artifact triển khai
    value: >-
      export(format="pt") cho một checkpoint đã đóng gói, export(format="onnx")
      cho một đồ thị QDQ INT8
    mono: true
verification: >-
  Đọc từ libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py và docs/quantization.md trên nhánh dev. Các
  con số về kích thước checkpoint là giá trị đo được ghi lại trong
  docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Thay cấu trúc rồi hiệu chuẩn. calib là một tập ảnh nhỏ KHÔNG NHÃN,

        # chỉ đọc theo chiều forward để suy ra dải giá trị và scale của
        activation

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # cùng các validator như một mô
        hình float

        qmodel.save("LibreYOLO9s-int8.pt")     # checkpoint mang theo một
        manifest quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Tham số
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # đường dẫn data.yaml hoặc tên có sẵn; None bỏ qua hiệu chuẩn
            samples=128,               # số ảnh hiệu chuẩn tối đa
            batch=8,                   # kích thước batch khi hiệu chuẩn
            algorithm="auto",          # auto và minmax là như nhau; percentile là lựa chọn còn lại
            keep_high_precision=None,  # None dùng chính sách của họ mô hình
            verbose=True,
        )
  reload:
    - label: Một checkpoint đã lượng tử hóa tải lại đúng như vậy
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Manifest quant dựng lại cấu trúc đã lượng tử hóa và các scale
        # trước khi trọng số được tải
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT chỉ là train() bình thường trên một mô hình đã lượng tử hóa
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Đây là một lượt tinh chỉnh, không phải chạy từ đầu: dùng learning rate
        của tinh chỉnh

        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD thêm vào các tham số distillation sẵn có
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Checkpoint PyTorch đã đóng gói
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Ghi ra LibreYOLO9s-int8-final.pt: trọng số và scale ít bit đã đóng
        gói,

        # các master fp32 bị loại bỏ, phần chưa lượng tử hóa còn lại được ép về
        fp16

        qmodel.export(format="pt")


        # remainder="fp32" giữ nguyên chính xác các tensor chưa lượng tử hóa

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Các cặp QuantizeLinear/DequantizeLinear ngay trong đồ thị, mang chính
        # các scale đã hiệu chuẩn hoặc đã huấn luyện bằng QAT của mô hình
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Quay lại float, giữ nguyên trọng số đã huấn luyện bằng QAT'
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # Giờ mọi bộ xuất float đều dùng được, ở bất kỳ precision nào nó hỗ trợ
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Cài đặt

Quantization không cần gói phụ thuộc thêm nào. Việc thay module, lượt hiệu chuẩn
và phép tính mô phỏng đều chạy trong PyTorch, nên `pip install libreyolo` là toàn
bộ yêu cầu. Các artifact triển khai cần đúng những gì định dạng của riêng chúng
cần, với đường ONNX thì đó là `libreyolo[onnx]`.

## Lượng tử hóa

<code-tabs name="quantize" />

`quantize()` biến đổi mô hình đã tải ngay tại chỗ rồi trả về chính nó. Không có
gradient nào tham gia: bước thay thế cài đặt các module đã lượng tử hóa, còn lượt
hiệu chuẩn chỉ chạy theo chiều forward.

Checkpoint thu được là một checkpoint LibreYOLO bình thường có gắn kèm một
manifest `quant`, nên nó tải lại với cấu trúc và các scale còn nguyên vẹn:

<code-tabs name="reload" />

Các checkpoint mà trainer ghi ra trong một lượt QAT cũng mang manifest đó, nghĩa
là `best.pt` của một lượt chạy như vậy bản thân nó đã là một checkpoint đã lượng
tử hóa.

## Các recipe

Bốn họ mô hình được hỗ trợ: `yolo9`, `rfdetr`, `birefnet` và `feynobg`.

| Recipe | Nó làm gì | Họ mô hình | Hiệu chuẩn |
|---|---|---|---|
| `fp16` | Ép kiểu về half precision với hợp đồng đầu vào và đầu ra float32. Chỉ dùng cho suy luận (inference). | cả bốn | không |
| `bf16` | Ép kiểu về bfloat16, vốn giữ nguyên dải số mũ của float32. Đây là cách sửa khi fp16 tràn số trên một mô hình kiểu DETR. Chỉ dùng cho inference. | cả bốn | không |
| `fp8` | Trọng số và activation E4M3 trên `Conv2d` và `Linear`: scale trọng số theo từng kênh, scale activation theo từng tensor đã hiệu chuẩn. | cả bốn | bắt buộc |
| `int8` | W8A8 trên `Conv2d` và `Linear`: trọng số đối xứng theo từng kênh, activation affine theo từng tensor. | cả bốn | bắt buộc, hoặc `calib=None` nếu chỉ lượng tử hóa trọng số |
| `w4a16` | Trọng số INT4 đối xứng theo nhóm, nhóm 128 dọc theo `in_features`, activation float, trên `Linear`. | rfdetr, birefnet, feynobg | không cần |
| `w4a8` | Trọng số INT4 theo nhóm cộng với activation INT8 đã hiệu chuẩn, trên `Linear`. | rfdetr, birefnet, feynobg | bắt buộc |
| `nvfp4` | W4A4 NVFP4 trên `Linear`: phần tử E2M1, khối 16 phần tử, scale khối FP8 E4M3, scale tensor FP32. Scale activation động. | rfdetr, birefnet, feynobg | không cần |
| `mxfp4` | OCP MXFP4 trên `Linear`: phần tử E2M1, khối 32 phần tử, scale khối E8M0 là lũy thừa của hai. Scale activation động. | rfdetr, birefnet, feynobg | không cần |
| `int2` | Chỉ dành cho nghiên cứu: trọng số 2 bit theo nhóm, nhóm 64, cộng với activation INT8, trên `Linear`. Chỉ lượng tử hóa sau huấn luyện thôi thì không dùng được, nên bắt buộc phải có QAT hoặc QAD. | rfdetr | bắt buộc |

Các recipe dưới 8 bit nhắm tới `nn.Linear` và bị từ chối với `yolo9` một cách có
chủ đích: trên phần cứng hiện nay việc tăng tốc đó chỉ có với GEMM, nên các
convolution vẫn nằm ở precision cao hơn. YOLO9 dùng `int8` hoặc `fp8`. `int2` bị
từ chối với `birefnet` và `feynobg` vì hai họ mô hình đó chỉ dùng cho inference,
nên phần QAT giúp hồi phục mà recipe này phụ thuộc vào lại không có ở đó.

Giá trị mặc định của từng họ mô hình giữ lớp đầu tiên và các head ở dạng float,
còn convolution DFL của YOLO9 thì không bao giờ bị lượng tử hóa: nó là một toán
tử kỳ vọng tích phân cố định. Ghi đè bằng `keep_high_precision=("head.",)` khi
bạn có lý do để làm vậy.

## Dữ liệu hiệu chuẩn không phải dữ liệu huấn luyện

`calib=` nhận vài trăm ảnh, không đọc nhãn nào, và chỉ chạy theo chiều forward để
ước lượng dải giá trị activation. `data=` trong `train()` và `val()` là tập dữ
liệu (dataset) có nhãn dùng cho gradient và các chỉ số. Đó là hai tham số khác
nhau với mục đích khác nhau, và mặc định của `calib` là `coco128.yaml`.

`algorithm="minmax"` giữ các cực trị tuyệt đối quan sát được qua các batch hiệu
chuẩn và là thứ mà `"auto"` chọn. `"percentile"` dùng trung bình của phân vị 0.1
và 99.9 trên từng batch; nó đã được đo là làm sụp đổ độ chính xác của họ mô hình
DETR, bởi các giá trị ngoại lai trong activation của transformer là thành phần
chịu lực. Thứ thực sự khắc phục độ nhạy INT8 của mô hình nhỏ là hiệu chuẩn trên
đủ số batch: với mặc định `coco128`, YOLO9-t rơi vào khoảng một điểm mAP so với
điểm số float của nó. Thuật toán được chọn sẽ được ghi lại trong manifest của
checkpoint.

## Lấy lại độ chính xác

<code-tabs name="train" />

Các module đã lượng tử hóa vẫn giữ trọng số master fp32 và áp dụng fake
quantization với một straight-through estimator, nên gradient vẫn tới được các
master và những trainer sẵn có hoạt động không cần đổi gì: EMA, AMP, việc tiếp
tục huấn luyện từ checkpoint và các tham số distillation đều kết hợp được với
nhau.

QAT là một lượt tinh chỉnh (fine-tuning) trên một mô hình đã được huấn luyện. Hãy
dùng learning rate của tinh chỉnh thay vì các giá trị mặc định dành cho huấn
luyện từ đầu, nếu không thì chỉ một lượt chạy ngắn cũng sẽ phá hỏng trọng số được
huấn luyện sẵn (pretrained), bất kể có quantization hay không. QAD có sẵn hay
không thì đi theo mức hỗ trợ distillation của từng họ mô hình, hiện tại nghĩa là
`yolo9` và `rfdetr`.

Các mô hình đã lượng tử hóa bằng `fp16` và `bf16` chỉ dùng cho inference, và
trainer từ chối chúng kèm một chỉ dẫn tới `amp=True`.

## Xuất

<code-tabs name="export" />

`format="pt"` kết tinh mô hình. Trọng số và scale ít bit đã đóng gói thay thế cho
các master, còn phần chưa lượng tử hóa còn lại được ép về fp16 trừ khi truyền
`remainder="fp32"`. Bất biến của việc đóng gói là khi giải nén sẽ tái tạo lại kết
quả mô phỏng đúng từng bit trên chính thiết bị bạn đã hoàn thiện mô hình, nên tệp
đã hoàn thiện đạt đúng điểm số bạn đã đánh giá. Số đo được: YOLO9-s int8 giảm từ
29.5 MB xuống 9.6 MB, RF-DETR-n nvfp4 từ 122 MB xuống 26 MB. Tải một tệp như vậy
sẽ cho ra một mô hình sẵn sàng cho inference, và gọi `train()` trên nó sẽ tự động
dựng lại các master từ trọng số đã đóng gói.

`format="onnx"` áp dụng cho các mô hình `int8` và phát ra một đồ thị QDQ mang
chính các scale đã hiệu chuẩn hoặc đã huấn luyện bằng QAT của mô hình, thứ mà
ONNX Runtime và TensorRT chạy bằng kernel INT8 thật. Đây là một đường đi khác với
[`export(format="onnx", int8=True)`](/docs/export/onnx) trên một mô hình float,
nơi ONNX Runtime tự suy ra các scale.

Các recipe ép kiểu thì hoàn toàn không cần bộ xuất dành cho mô hình đã lượng tử
hóa:

<code-tabs name="dequantize" />

## Ràng buộc

Phép tính đã lượng tử hóa được thực thi ở dạng mô phỏng, tức fake quantization
tính trong các ốc đảo float32 ngay cả khi có AMP. Mô phỏng đúng về mặt số học,
nên một điểm `val()` trên bất kỳ thiết bị nào cũng là một khẳng định thật về phép
tính đã lượng tử hóa. Nó không phải một khẳng định về tốc độ.

Hai ngoại lệ được thực thi natively. `fp16` và `bf16` là các phép ép kiểu thông
thường. Các module `fp8` đã hoàn thiện chạy GEMM của chúng trực tiếp trên trọng
số E4M3 đã đóng gói thông qua `torch._scaled_mm` trên phần cứng lớp Ada, Hopper
và Blackwell, dùng chính các scale activation đã hiệu chuẩn như trong mô phỏng;
đặt `LIBREYOLO_KERNELS=off` sẽ khôi phục đúng đường mô phỏng ở mọi nơi.

Phạm vi triển khai hẹp hơn danh sách recipe. Ở đây chỉ `int8` có một dạng ONNX
triển khai được; `fp8` và các recipe dưới 8 bit cho lớp linear chạy trong PyTorch
và kết tinh qua `format="pt"`. Yêu cầu xuất ONNX từ chúng sẽ báo lỗi kèm đúng chỉ
dẫn đó, và yêu cầu bất kỳ định dạng nào không phải ONNX từ một mô hình `int8`
cũng vậy: hãy dựng các engine phía sau từ đồ thị QDQ.

Xuất một mô hình `int8` mà activation của nó chưa từng được hiệu chuẩn sẽ ghi ra
một cảnh báo và tạo ra một đồ thị chỉ mang phần lượng tử hóa trọng số.
