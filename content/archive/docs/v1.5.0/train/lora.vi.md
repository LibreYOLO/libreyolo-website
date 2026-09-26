---
title: Tinh chỉnh LoRA
seo_title: Tinh chỉnh LoRA trong LibreYOLO
description: >-
  Tinh chỉnh transformer detector với ít VRAM bằng lora=True. Chín family nào hỗ
  trợ, recipe adapter theo từng family và cách checkpoint hoạt động.
lead: >-
  LoRA đóng băng các phần nặng đã được huấn luyện sẵn của mô hình và huấn luyện
  những adapter low-rank nhỏ bên cạnh chúng, cùng các lớp phải giữ dense. Trong
  LibreYOLO, toàn bộ interface công khai chỉ là một giá trị boolean.
keywords:
  - tinh chỉnh lora
  - fine tuning ít tham số
  - peft
  - dora
  - huấn luyện ít vram
  - rf-detr lora
  - d-fine lora
  - gộp adapter
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: Xuất sẽ gộp các adapter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Gộp tại chỗ
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Cài đặt

LoRA dựa trên dependency tùy chọn `peft`.

<code-tabs name="install" />

Nếu thiếu dependency này, `lora=True` sẽ phát sinh `ImportError` nêu rõ lệnh đó,
thay vì vô tình huấn luyện bằng lượt tinh chỉnh đầy đủ.

## Cách dùng

<code-tabs name="train" />

`lora=True` là toàn bộ interface. Rank, alpha, dropout và module đích được cố
định theo từng family để khớp với tham chiếu upstream tương ứng, không phải là
các nút điều chỉnh dành cho người dùng.

Family không hỗ trợ LoRA sẽ phát sinh lỗi khi thiết lập thay vì bỏ qua flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLI từ chối sớm hơn, trước khi mô hình được dựng, bằng allowlist riêng chứa cùng
chín family.

## Các family được hỗ trợ

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 và v4, EC và ConvNeXt. Cổng kiểm
tra là thuộc tính `supports_lora` trên class trainer của từng family, còn CLI có
một allowlist tương ứng.

Phạm vi tác vụ hẹp hơn phạm vi family. D-FINE và EC chỉ hỗ trợ phát hiện, đường
dẫn phân đoạn và tư thế của chúng sẽ phát sinh lỗi. Đường dẫn ngữ nghĩa của
RF-DETR cũng phát sinh lỗi. ConvNeXt dành cho phân loại.

Mọi trường hợp khác đều phát sinh lỗi. Không có chế độ một phần hoặc im lặng.

## Mỗi recipe làm gì

Các recipe khác nhau vì kiến trúc khác nhau, và một recipe hoạt động trên
backbone ViT sẽ không có gì để gắn vào trên backbone tích chập.

RF-DETR dùng DoRA, tức LoRA phân rã trọng số, ở rank 16 và alpha 16 trên các phép
chiếu attention `query`, `key` và `value` của backbone DINOv2, khớp với tham
chiếu RF-DETR. Backbone ViT bị đóng băng; projector, decoder và detection head
tiếp tục huấn luyện bình thường.

D-FINE, DEIM và RT-DETR v1, v2 và v4 ghép backbone tích chập với hybrid encoder
transformer và deformable decoder, nên điểm phân tách thay đổi. Backbone tích
chập bị đóng băng hoàn toàn, nhờ đó cũng bỏ qua backward pass của nó. Các block
transformer đóng băng trọng số cơ sở và huấn luyện adapter LoRA thuần ở cùng
rank 16 và alpha 16 trên các lớp tuyến tính: feed-forward `linear1` và `linear2`,
gate và các phép chiếu deformable attention. Mọi phần khác, gồm fusion tích chập
của encoder, phép chiếu đầu vào, prediction head và query embedding, tiếp tục
được huấn luyện theo dạng dense.

Hai chi tiết trong recipe đó là có chủ ý. Self-attention của decoder vẫn đóng
băng mà không có adapter vì `nn.MultiheadAttention` của PyTorch đọc trực tiếp
`out_proj.weight` và sẽ âm thầm bỏ qua adapter được chèn. Đồng thời, recipe dùng
LoRA thuần thay vì DoRA vì một số lớp tuyến tính của decoder được khởi tạo bằng
0 theo thiết kế và phép chuẩn hóa độ lớn của DoRA sẽ chia cho norm của trọng số.

DEIMv2 dùng cùng recipe với các lớp feed-forward SwiGLU `w12` và `w3` làm đích.
Các kích thước S, M, L và X của nó còn có backbone ViT DINOv3. Phần cơ sở ViT bị
đóng băng và các lớp attention `qkv` đã hợp nhất nhận adapter, trong khi pyramid
tích chập Spatial Tuning Adapter tiếp tục huấn luyện như thành phần tương tự
projector. Các adapter `qkv` đó được chèn ngay cả khi cấu hình phân phối ViT ở
trạng thái đóng băng vì mục đích chính là điều chỉnh một backbone đã đóng băng.
Các kích thước nhỏ hơn S dùng backbone tích chập và áp dụng recipe thuần.

EC là một DETR có backbone ViT được bao quanh bởi pyramid projector tích chập có
thể huấn luyện. Phần cơ sở ViT bị đóng băng và các lớp `qkv` nhận adapter, các
block transformer áp dụng recipe chung, còn projector và head giữ dạng dense.

Các block ConvNeXt có MLP tuyến tính channels-last là `fc1` và `fc2`, và các lớp
này nhận adapter thuần. Depthwise convolution, norm và tham số layer-scale bị
đóng băng. Classification head giữ dạng dense để số lượng lớp đối tượng tùy
chỉnh tiếp tục hoạt động.

Detection head và classification head luôn có thể huấn luyện trong mọi recipe
vì số lượng lớp đối tượng tùy chỉnh cần một head mới được huấn luyện.

## Checkpoint và xuất

`best.pt` và `last.pt` giữ lại các tensor adapter, vì vậy một lượt chạy LoRA có
thể tiếp tục hoặc được kiểm tra giống mọi lượt chạy khác. Việc nạp một trong các
checkpoint này cần cài thành phần bổ sung `lora` vì loader thực hiện lại thao tác
chèn adapter để các key khớp nhau.

`export()` gộp adapter vào trọng số dense, vì vậy artifact đã xuất không phụ
thuộc vào `peft`. Thao tác gộp tương tự cũng có thể dùng trực tiếp cho mô hình
trong bộ nhớ.

<code-tabs name="merge" />

Sau khi gộp, cây module hoàn toàn ở dạng dense và lần gộp thứ hai không làm gì.

## Nội dung được tiết kiệm và không được tiết kiệm

LoRA giảm bộ nhớ cho optimizer và gradient, và trên các family đóng băng hẳn
backbone, nó cũng bỏ qua backward pass của backbone đó.

Bộ nhớ activation không thay đổi. Forward activation vẫn phải được giữ lại cho
bất cứ phần nào còn có thể huấn luyện, và đây thường là yếu tố quyết định mức
đỉnh. Với giới hạn VRAM chặt nhất, hãy giảm cả `batch` hoặc `imgsz`.

## Nội dung liên quan

- Xem [Đóng băng lớp](/docs/train/layer-freezing) để biết cách khác nhằm huấn
  luyện một tập con của trọng số, áp dụng cho mọi family và không cần dependency
  bổ sung. `freeze` và `lora=True` kết hợp được với nhau: tham số adapter vẫn có
  thể huấn luyện ngay cả khi nhóm backbone cha bị đóng băng.
- Xem [Siêu tham số](/docs/train/hyperparameters) để biết về `batch`, `imgsz` và
  phần còn lại của `train()`.
