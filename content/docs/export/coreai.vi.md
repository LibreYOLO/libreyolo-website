---
title: Core AI
seo_title: Xuất sang Apple Core AI từ LibreYOLO
description: >-
  Xuất một mô hình LibreYOLO thành asset .aimodel của Apple Core AI: chỉ chạy
  trên macOS, canvas cố định, FP32, và hợp đồng về thứ tự các đầu ra có tên mà
  bên tiêu thụ phải tuân theo.
lead: >-
  Core AI là stack suy luận (inference) trên thiết bị của Apple. LibreYOLO bắt
  đồ thị (capture) mô hình bằng torch.export, đưa nó xuống qua bộ chuyển đổi
  Core AI, rồi ghi ra một asset .aimodel mang metadata của mô hình và tên các
  đầu ra đã xuất.
keywords:
  - xuất libreyolo sang core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - suy luận trên thiết bị apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Kết quả ghi ra
    value: Một asset .aimodel có gắn kèm metadata
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Tải lại
    value: Không qua LibreYOLO. Bên tiêu thụ dùng trực tiếp runtime của Core AI.
  - label: Hình dạng
    value: Canvas cố định. dynamic=True ném NotImplementedError.
  - label: Precision
    value: Chỉ FP32. half=True và int8=True bị từ chối.
  - label: Yêu cầu
    value: >-
      macOS. Toolchain không chuyển đổi cũng không chạy ở nơi khác, và
      coreai-torch ghim torch vào 2.11.x.
verification: >-
  Đọc từ libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py và pyproject.toml
  trên nhánh dev.
snippets:
  install:
    - label: 'Cài đặt, trên macOS'
      language: bash
      code: |
        # Cố ý để ngoài mọi extra tổng hợp: coreai-torch ghim torch
        # vào 2.11.x và sẽ kéo cả môi trường sang phiên bản đó
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Ghi ra weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Tham số
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, hoặc (cao, rộng); đây là canvas khi chạy
            batch=1,
            output_path=None, # None ghi ra weights/<stem>.aimodel
        )

        # dynamic=True ném NotImplementedError
        # half=True và int8=True bị từ chối trong lúc kiểm tra hợp lệ
  outputs:
    - label: Đọc thứ tự đầu ra trước khi đấu nối một bên tiêu thụ
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")

        model.export(format="coreai", imgsz=640)


        # Metadata của asset ghi lại tên các đầu ra đã xuất, theo thứ tự đồ thị,

        # dưới khóa "coreai_output_names". Hãy map dictionary mà Core AI trả về

        # theo tên bằng danh sách đó; đừng bao giờ ghép theo vị trí với tuple
        eager
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi xuất
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Cài đặt

Định dạng này chỉ dành cho macOS. Yêu cầu `coreai-torch` mang marker
`sys_platform == 'darwin'`, và toolchain không chuyển đổi cũng không chạy ở bất
kỳ nơi nào khác.

<code-tabs name="install" />

Extra này nằm ngoài mọi extra tổng hợp, kể cả `libreyolo[all]`, vì
`coreai-torch` ghim torch vào dòng 2.11. Hãy cài nó vào một môi trường mà bạn
sẵn sàng ràng buộc theo đúng cặp đó.

## Xuất mô hình

<code-tabs name="export" />

Capture ở đây là `torch.export`, một lần bắt đồ thị thật sự có guard, chứ không
phải một trace đơn lẻ được ghi lại. Cách này nghiêm ngặt hơn đường Core ML: việc
đọc scalar trên host và luồng điều khiển phụ thuộc dữ liệu bị từ chối thay vì bị
bake cứng vào một cách âm thầm, và đó là lý do vài họ mô hình bị chặn ở đây kèm
một lỗi capture được ghi lại.

Ba bước chuẩn bị chạy bên trong một scope khôi phục lại mô hình đang sống của
phía gọi, bất kể việc xuất thành công hay thất bại. Các họ bắt nguồn từ Darknet
được gộp batch normalization ở chế độ inference vào đúng các convolution đứng
trước, vì Core AI 0.4.1 không giữ được công thức epsilon-sau-căn-bậc-hai của
Darknet. Các họ dùng grid và anchor được đóng băng anchor cho canvas cố định.
RF-DETR được bake lại position embedding cho canvas yêu cầu bằng cách chạy lại
chính đường bake của mô hình, vì bộ chuyển đổi không có lowering cho
`aten._upsample_bicubic2d_aa`.

Bước lowering gộp phần decomposition tham chiếu của PyTorch cho
`aten.grid_sampler_2d` vào bảng decomposition, vì bộ chuyển đổi Core AI không có
lowering cho bộ sampler deformable attention mà các họ DETR dùng.

Asset khai báo OS tối thiểu là v27, giá trị duy nhất mà toolchain đưa ra. Điều
đó chặn ở khâu triển khai chứ không phải khâu chuyển đổi: việc chuyển đổi và
việc chạy từ phía Python vẫn hoạt động trên macOS cũ hơn nhờ runtime nằm trong
wheel, nhưng số học khác nhau giữa các phiên bản OS, nên độ khớp số học (parity)
được ghi nhận là đo trên macOS 27.

## Chạy artifact

Không có mục Core AI nào trong `libreyolo/backends`, nên `LibreYOLO()` không tải
một `.aimodel`. Bên tiêu thụ dùng trực tiếp runtime của Core AI, còn tiền xử lý,
giải mã, NMS và việc scale lại tọa độ là phần việc của họ. Một dòng đã được xác
nhận trong ma trận hỗ trợ là lời khẳng định rằng đồ thị đã xuất tính ra đúng
những con số như bản tham chiếu, chứ không phải rằng `predict` sẽ chạy được nó.

Thứ duy nhất mà một bên tiêu thụ không thể tự suy ra lại là thứ tự các đầu ra:

<code-tabs name="outputs" />

Core AI trả về một dictionary có tên, và thứ tự khóa của nó không khớp với thứ
tự tuple của forward ở chế độ eager, cũng không khớp với bất cứ thứ gì đoán
được. Tên các đầu ra đã xuất được ghi vào metadata của asset dưới khóa
`coreai_output_names` chính vì lý do này. Hãy map theo tên.

## Ràng buộc

Canvas cố định, FP32, batch đúng như lúc xuất. `dynamic=True` ném
`NotImplementedError`, còn `half=True` và `int8=True` bị từ chối trong lúc kiểm
tra hợp lệ.

Độ phủ ở phía chuyển đổi là rộng. Các tổ hợp đã được xác nhận gồm các họ YOLO9,
YOLOX, YOLO7, bốn bộ phát hiện thời Darknet, YOLO-NAS, PicoDet, RTMDet, RT-DETR,
RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC và phát hiện đối tượng RF-DETR;
bốn họ phân loại CNN cộng thêm CLIP và SigLIP2 với lớp đối tượng cố định; Depth
Anything V2 và ZipDepth; phục hồi ảnh với NAFNet và Real-ESRGAN; phân đoạn ngữ
nghĩa với PIDNet và LingBotVision; và phát hiện điểm FOMO. Mỗi tổ hợp mang theo
bối cảnh riêng đã được ghi lại, thứ mà `libreyolo formats` in ra.

Bị chặn, kèm lý do được ghi lại cho từng tổ hợp:

| Tổ hợp | Lý do |
|---|---|
| EoMT phân đoạn ngữ nghĩa | Capture nghiêm ngặt thất bại với `GuardOnDataDependentSymNode`: có chỗ trong đường xử lý mặt nạ (mask) đọc một giá trị từ tensor rồi rẽ nhánh theo nó |
| SegFormer phân đoạn ngữ nghĩa | Đường capture chưa được đánh giá, và trọng số công bố của nó là phi thương mại bất kể định dạng nào |
| L2CS ước lượng hướng nhìn | Bản thân mô hình chỉ hỗ trợ ONNX, TorchScript, ExecuTorch, TensorRT và OpenVINO, đây là quyết định từ phía mô hình |
| Depth Anything 3 ước lượng độ sâu | Họ này từ chối xuất với mọi định dạng |

RF-DETR có một lưu ý đáng đọc trước khi so sánh các artifact. Parity của nó được
ghi nhận so với đồ thị mà chính bộ xuất Core AI chuẩn bị, chứ không phải so với
ONNX, và ở canvas 640 thì artifact ONNX của RF-DETR lệch so với đồ thị đã chuẩn
bị đó. Lần bake lại của Core AI giữ nguyên phép resize có antialiasing mà mô
hình ở chế độ eager thực hiện, trong khi đường ONNX tắt antialiasing. Vì vậy
ONNX không phải là bản tham chiếu hợp lệ cho họ đó ở canvas không phải kích
thước gốc.

Với định dạng cũ hơn của Apple, xem [Core ML](/docs/export/coreml). Với toàn bộ
lưới họ mô hình và tác vụ, xem [ma trận xuất mô hình](/docs/reference/export-matrix).
Với một tổ hợp:

<code-tabs name="support" />
