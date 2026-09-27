---
title: Kernels
seo_title: Registry kernel LibreYOLO và kernel Hub
description: >-
  Cách LibreYOLO chọn bản triển khai tăng tốc: registry kernel trong
  libreyolo/kernels, kernel MS-deform-attn tùy chọn trên Hugging Face Hub và
  công tắc fused attention.
lead: >-
  Mọi thao tác tăng tốc trong LibreYOLO đều có bản mặc định khả chuyển và đôi
  khi có biến thể nhanh hơn được đăng ký bên trên. Việc lựa chọn diễn ra trong
  runtime theo predicate; thiếu dependency tùy chọn sẽ dùng luồng dự phòng thay
  vì lỗi; đồ thị đã xuất luôn dùng luồng khả chuyển.
keywords:
  - libreyolo kernels
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels extra
  - ms_deform_attn kernel
  - set_fused_attention
  - libreyolo triton kernels
last_verified: 1.5.0
verification: >-
  API registry được đọc từ libreyolo/kernels/__init__.py ở v1.5.0, API attention
  từ libreyolo/kernels/attention/__init__.py và sdpa.py, provider Hub từ
  libreyolo/kernels/attention/ms_deform_attn.py gồm revision được ghim và
  predicate đủ điều kiện. Bố cục thư mục được liệt kê từ libreyolo/kernels/.
  Định nghĩa extra lấy từ pyproject.toml. Ghi chú hành vi và số liệu benchmark
  lấy từ docs/kernels.md. Lịch sử điều kiện v1.4.0 lấy từ commit nối slot
  RF-DETR và mục CHANGELOG 1.5.0.
meta:
  - label: Package
    value: libreyolo.kernels
    mono: true
  - label: Extra tùy chọn tham gia
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Buộc dùng bản tham chiếu
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Xem lựa chọn hiện tại
      language: python
      code: >
        import libreyolo.kernels as kernels


        # Ánh xạ slot thao tác đến tên bản triển khai được chọn hoặc
        "unavailable".

        print(kernels.active())
    - label: Buộc dùng luồng tham chiếu
      language: bash
      code: |
        # off và reference có cùng ý nghĩa, đồng thời bỏ qua hoàn toàn
        # việc import các provider tăng tốc.
        LIBREYOLO_KERNELS=off python train.py
    - label: Tắt kernel Hub mà không gỡ cài đặt
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Chuyển một họ sang fused attention
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Trả về số module attention đã chuyển.
        print(set_fused_attention(model))
    - label: Đăng ký bản của bạn
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Registry

`libreyolo/kernels/` là registry runtime nhỏ gồm các bản triển khai có thể cắm
thêm. Slot thao tác là tên như `fake_quant_fp8` hoặc `ms_deform_attn`. Bên gọi
yêu cầu registry một slot và nhận bản triển khai đã đăng ký đầu tiên vượt qua
predicate, trong đó đăng ký mới nhất thắng; nếu không có dạng nào áp dụng thì
chuyển sang bản triển khai tham chiếu.

Cấu trúc này tồn tại để dependency tùy chọn không bao giờ là yêu cầu bắt buộc.
Máy không có Triton, CUDA hoặc gói `kernels` vẫn chạy cùng mã và tạo cùng số liệu,
chỉ chậm hơn.

| Hàm | Mục đích |
|---|---|
| `active()` | Ánh xạ slot thao tác đến tên bản triển khai được chọn hoặc `"unavailable"` |
| `resolve(op)` | Callable sẽ chạy hoặc `None` |
| `register(op, impl, *, name, predicate=None)` | Thêm bản triển khai, mới nhất đứng đầu |
| `unregister(op, name)` | Loại một bản |
| `clear_cache()` | Xóa kết quả phân giải đã ghi nhớ |

<code-tabs name="usage" />

Predicate phát sinh lỗi sẽ được bắt và cảnh báo, không bao giờ lan truyền, nên
bản triển khai bên thứ ba bị lỗi sẽ hạ xuống luồng khả chuyển thay vì làm hỏng dự đoán.

### Bố cục

Cây được tổ chức theo mục đích trước rồi đến backend, nên slot được tìm theo nội dung tính toán thay vì thư viện tình cờ triển khai nó hiện tại.

| Thư mục | Nội dung |
|---|---|
| `kernels/quant/simulate/` | Kernel Triton lượng tử hóa giả với backward straight-through trên mọi thiết bị. Dùng cho cả QAT và lượng tử hóa sau huấn luyện mô phỏng |
| `kernels/quant/execute/` | Luồng độ chính xác thực chỉ cho mô hình đã hoàn tất, không backward: GEMM tensor-core FP8, phần mở đầu và kết thúc Triton hợp nhất cùng các kernel giải nén trọng số đóng gói |
| `kernels/attention/` | Thao tác attention dùng chung giữa các họ: slot `ms_deform_attn` và chính sách fused-SDPA |

Ranh giới giữa `simulate` và `execute` là mô hình đã hoàn tất hay chưa, không
phải đang huấn luyện hay triển khai. Bản triển khai tham chiếu nằm trong
`libreyolo/quant/`, nơi định nghĩa ý nghĩa các số; `kernels/` chỉ làm chúng nhanh.
Đóng gói trọng số không có biến thể vì đó là giao diện checkpoint.

Slot GEMM và attention không có bản triển khai tham chiếu. Bên gọi phải kiểm tra
`resolve()` có trả về gì và duy trì luồng khả chuyển riêng, vì vậy đồ thị ONNX,
TensorRT và `torch.export` luôn chứa phép toán khả chuyển.

### Ghi đè lựa chọn

`LIBREYOLO_KERNELS=off` hoặc `=reference` buộc dùng bản triển khai tham chiếu và
bỏ qua hoàn toàn việc import provider tăng tốc. Mọi giá trị khác giới hạn lựa
chọn ở các bản triển khai đăng ký dưới tên đó. `LIBREYOLO_QUANT_KERNELS` được
tôn trọng làm bí danh cũ từ thời registry nằm trong `libreyolo/quant/` và chỉ
được đọc khi chưa đặt `LIBREYOLO_KERNELS`. Cả hai được liệt kê cùng các mục khác
trong [cài đặt](/docs/reference/settings).

## Kernel Hub

Kernel CUDA đã biên dịch được công bố trên Hugging Face Hub sẽ tải trong runtime
qua gói `kernels` tùy chọn. Không có gì được đóng gói vào LibreYOLO; artifact
được gói đó lấy và lưu vào bộ nhớ đệm, còn mỗi provider ghim commit revision đã
kiểm tra, nên nâng pin cần lượt kiểm tra tương đương trên GPU trước khi tích hợp.

Cài extra là thao tác tham gia:

```bash
pip install "libreyolo[hub-kernels]"
```

Khi không có gói, không có gì thay đổi và không gửi yêu cầu mạng.
`LIBREYOLO_HUB_KERNELS=0` tắt việc lấy dữ liệu mà không gỡ cài đặt. Kernel không
tải hoặc chạy được sẽ tự tắt trong phần còn lại của tiến trình và dùng luồng dự
phòng kèm một cảnh báo.

Hiện một slot được Hub hỗ trợ: `ms_deform_attn`, forward và backward deformable
attention multi-scale đã biên dịch từ Deformable DETR theo Apache 2.0. Nó được
nối vào toàn bộ dòng deformable: RF-DETR, Deformable DETR, DINO-DETR, LW-DETR,
Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM, DEIMv2, EC và
OV-DEIM. Vì backward cũng được biên dịch, huấn luyện và dự đoán đều hưởng lợi.

Điều kiện đủ tiêu chuẩn được cố ý thu hẹp. Đầu vào phải là CUDA và float32, còn
thực thi phải ở eager: provider từ chối dưới `torch.jit.is_tracing()`,
`torch.compiler.is_compiling()`, `torch.compiler.is_exporting()` và
`torch.onnx.is_in_onnx_export()`. Hai bố cục đầu vào cũng chuyển sang luồng khả
chuyển: số điểm theo cấp thay đổi giữa các cấp và lấy mẫu chỉ số nguyên rời rạc.
Biến thể tư thế EC chưa được nối.

### Kernel này mới có thể được dùng

Hãy đọc phần này trước khi cài extra trên dự án hiện có.

Trong v1.4.0, slot được tra cứu bên trong hàm trợ giúp, sau điều kiện yêu cầu
không có các cặp spatial-shape. RF-DETR luôn chuyển các cặp đó qua decoder, nên
điều kiện không bao giờ đúng và kernel không thực thi trong bất kỳ eager forward
nào. Vị trí tra cứu đã chuyển trong v1.5.0 và kernel hiện thực sự chạy.

Hệ quả thực tế là nâng lên v1.5.0 *và* cài `libreyolo[hub-kernels]` trên CUDA
khiến RF-DETR cùng dòng mô hình lần đầu lấy forward từ binary đã biên dịch. Do
đó, dự đoán và chỉ số có thể dịch chuyển trong dung sai số thực. Bản cài đặt mặc
định không có extra không bị ảnh hưởng. Nếu so sánh chỉ số qua bản nâng cấp, hãy
giữ extra cố định hoặc đặt `LIBREYOLO_HUB_KERNELS=0` ở cả hai phía.

## Fused attention

Fused scaled dot-product attention không cần dependency tùy chọn, chỉ cần
PyTorch tiêu chuẩn, nên được chi phối bằng chính sách thay vì tính khả dụng. Có
hai quy tắc áp dụng.

Thứ nhất, capture graph không bao giờ dùng nó. Mỗi vị trí lời gọi được thay vẫn
giữ phương trình thao tác primitive phía sau bước kiểm tra xuất, bao quát bản
xuất ONNX có opset mặc định không có symbolic SDPA và `torch.jit.trace` mà
TorchScript, CoreML cùng NCNN đều đi qua. Capture Dynamo được cố ý đặt ngoài cổng
vì `torch.compile` hạ SDPA tốt hơn phép toán thủ công, còn Core AI và ExecuTorch
tự phân rã SDPA thành ATen lõi.

Thứ hai, tiêu chuẩn tương đương để đặt làm mặc định là chính xác từng byte. Các
họ vượt qua dùng SDPA theo mặc định: SegFormer, Depth Anything và MoGe-2, BERT,
Grounding DINO, SwinIR và PP-OCR. Các họ không vượt qua giữ phép toán thủ công và
cung cấp cờ `fused_attn`, được `set_fused_attention(model)` chuyển: Swin, backbone
Swin của DINO-DETR, BiRefNet và FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth và
MobileSAM. ViT và DeiT có cùng cờ nhưng mặc định bật theo thượng nguồn, nên cùng
lời gọi với `enabled=False` sẽ tắt.

Tính năng đáng dùng ở nơi áp dụng. Trên RTX 5070 Ti dưới fp16 autocast, window
attention của Swin giảm từ 1.278 ms xuống 0.721 ms, nhanh hơn 1.77 lần, còn vision
attention của OWLv2 giảm từ 6.483 ms xuống 1.735 ms, nhanh hơn 3.74 lần.

## Phần cứng

| Nền tảng | Hành vi |
|---|---|
| CPU và MPS | Mọi predicate CUDA và Triton đều thất bại, nên tất cả chạy bản tham chiếu |
| NVIDIA CUDA | Kernel Triton cùng kernel Hub và GEMM đủ điều kiện được kích hoạt |
| AMD ROCm | Triton có thể kích hoạt vì wheel ROCm phân phối backend AMD của Triton, nhưng mức tương đương chỉ được kiểm tra trên NVIDIA trong CI |

## Thêm bản triển khai

Gọi `register()` với tên và predicate. Kernel đã biên dịch ngoài cây có thể phân
phối dưới dạng gói `libreyolo_kernels` riêng tự đăng ký khi import, nhờ đó backend
riêng hoàn toàn nằm ngoài cây LibreYOLO.

Mức tương đương là cổng cho mọi thành phần trong cây: forward khớp chính xác với
bản tham chiếu và gradient nằm trong 1e-6 của estimator straight-through trên
tập shape do bộ kiểm thử cung cấp.

Lựa chọn kernel tương tác với [CUDA graph](/docs/reference/cuda-graphs): ma trận
tương đương suy luận chạy khi chưa cài gói `kernels`, nên không bao quát độ an
toàn capture khi kernel đã biên dịch hoạt động.

