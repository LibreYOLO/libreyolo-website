---
title: RKNN
seo_title: Xuất sang RKNN cho NPU Rockchip
description: >-
  Biên dịch một mô hình phát hiện đối tượng LibreYOLO thành tệp .rknn của
  Rockchip: SDK của hãng mà bạn tự cài, bốn biến thể RK3588 đã được kiểm chứng,
  và việc đối chiếu số học trên trình mô phỏng.
lead: >-
  RKNN là định dạng NPU đã biên dịch của Rockchip. LibreYOLO xuất một ONNX trung
  gian opset-19, biên dịch nó bằng SDK RKNN Toolkit2, và có thể so sánh đồ thị
  đã biên dịch với ONNX Runtime trong trình mô phỏng chạy trên máy chủ của
  Toolkit2 mà không cần board.
keywords:
  - xuất yolo sang rknn
  - npu rockchip
  - rk3588
  - rknn-toolkit2
  - đối chiếu số học rknn simulator
  - chạy yolo trên orange pi rockchip
last_verified: 1.5.0
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Kết quả ghi ra
    value: >-
      Một tệp .rknn, một tệp sidecar .rknn.metadata.json, và một báo cáo
      .rknn.parity.json khi verify=True
  - label: Phụ thuộc thêm
    value: 'Không có gì trên PyPI. rknn-toolkit2 là SDK của hãng, bạn tự cài lấy.'
  - label: Tải lại
    value: >-
      Không qua LibreYOLO. Tệp đã biên dịch chạy trên board bằng runtime của
      Rockchip.
  - label: Hình dạng
    value: 'Vuông cố định, batch 1, opset 19. Cả ba đều bị bắt buộc.'
  - label: Precision
    value: Bản dựng dấu phẩy động của hãng. half=True và int8=True đều bị từ chối.
  - label: Phạm vi
    value: >-
      Bốn biến thể phát hiện đối tượng trên RK3588: YOLO9-t, YOLO9-E2E-t,
      PicoDet-s và YOLO-NAS-s
verification: >-
  Đọc từ libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py và docs/rknn.md trên nhánh dev. Các số liệu đối
  chiếu đo được lấy từ bản ghi kiểm chứng ngày 2026-08-04 trong docs/rknn.md.
snippets:
  install:
    - label: Phía LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK của hãng, bạn tự cài'
      language: bash
      code: |
        # rknn-toolkit2 là SDK của Rockchip theo một giấy phép riêng. LibreYOLO
        # không đóng gói cũng không cài nó. Chỉ chạy trên Linux x86_64; trên
        # Windows hãy dùng WSL2 hoặc một container Linux.
        #
        # Toolkit2 2.3.2 cần setuptools<81 và lỗi với ONNX 1.19 trở lên, vì bản
        # này đã bỏ onnx.mapping mà trình biên dịch của nó vẫn import
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Sau đó cài wheel rknn-toolkit2 tương ứng từ kho wheel riêng của
        # Rockchip, rồi xác nhận nó import được:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Ghi ra weights/LibreYOLO9t.rknn và
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Tham số
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # target platform; target= và target_platform= cũng dùng được
            imgsz=640,         # phải khớp với canvas đã ghi nhận của biến thể
            batch=1,           # mọi giá trị khác đều ném NotImplementedError
            dynamic=False,     # True ném ValueError
            opset=19,          # mọi giá trị khác đều ném NotImplementedError
            verify=False,      # True chạy trình mô phỏng PC và chặn theo ngưỡng parity
        )
  parity:
    - label: Đối chiếu không cần board với một tệp ONNX có sẵn
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Kiểm tra một họ mô hình và một tác vụ trước khi biên dịch
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Cài đặt

Việc biên dịch cần RKNN Toolkit2 của Rockchip, thứ được phát hành như một SDK của
hãng theo giấy phép riêng của Rockchip và không phải là một phụ thuộc của
LibreYOLO. Không có extra `libreyolo[rknn]`, và không phần nào của định dạng này
cài được bằng một dòng lệnh duy nhất.

<code-tabs name="install" />

Không cần board để biên dịch hay để kiểm tra độ khớp số học (parity). Cần một
board RK3588 để đo độ trễ, điện năng và nhiệt, và chưa có phép đo nào trong số đó
được ghi nhận.

## Xuất

<code-tabs name="export" />

Yêu cầu xuất được kiểm tra đối chiếu với một danh sách các biến thể mô hình chính
xác trước khi bất cứ thứ gì được biên dịch, và khung ảnh (canvas) cũng được kiểm
tra: truyền một `imgsz` khác với giá trị mà biến thể đó đã được ghi nhận sẽ ném
lỗi thay vì lặng lẽ biên dịch một thứ chưa được kiểm thử. LibreYOLO ghi ra một
ONNX trung gian opset-19, biên dịch nó, tùy chọn mô phỏng nó, rồi xóa tệp trung
gian sau đó.

Metadata nằm trong một tệp sidecar tên `<model>.rknn.metadata.json`, vì định dạng
RKNN không có trường metadata mang theo được.

`verify=True` chạy trình mô phỏng PC của Toolkit2 ngay trong phiên đã biên dịch ra
tệp đó, so sánh từng đầu ra với ONNX Runtime trên cùng một đầu vào, và ghi ra
`<model>.rknn.parity.json` kèm các chỉ số sai số cho từng đầu ra. Ngưỡng đặt ra là
cosine similarity tối thiểu 0.9999 và RMSE chuẩn hóa tối đa 0.02, áp dụng cho mọi
đầu ra chưa khớp sẵn theo từng phần tử; bản dựng dấu phẩy động của hãng hạ các
tensor nội bộ xuống half precision, nên `allclose` chặt chẽ không đúng ngay cả khi
các bounding box đã giải mã vẫn ổn định. Một lần chạy thất bại sẽ ghi ra
`<model>.rknn.failed.parity.json`, loại bỏ ứng viên, và giữ nguyên bản xuất thành
công trước đó tại đường dẫn ấy.

Để so sánh một tệp ONNX bạn đã có sẵn mà không cần xuất lại:

<code-tabs name="parity" />

Trình mô phỏng của Toolkit2 chạy đồ thị trong bộ nhớ do `load_onnx` và `build` tạo
ra. Nó không thể nạp lại một tệp `.rknn` gắn với một target cụ thể khi không có
board, và đó là lý do `verify=True` thực hiện biên dịch, xuất và mô phỏng trong
cùng một phiên.

## Chạy tệp đã xuất

Không có mục RKNN nào trong `libreyolo/backends`, nên `LibreYOLO()` không nạp tệp
`.rknn`. Tệp đã biên dịch được triển khai lên board và chạy bằng runtime riêng của
Rockchip, và ở đó tiền xử lý, giải mã, NMS cùng việc quy đổi lại tọa độ là phần
việc của ứng dụng.

`<model>.rknn.metadata.json` mang theo tên các lớp đối tượng, kích thước đầu vào,
tác vụ và target platform, đúng những gì một ứng dụng cần để tái hiện phần hậu xử
lý của LibreYOLO. Hãy đóng gói nó kèm mô hình đã biên dịch.

Để kiểm tra ở phía máy chủ mà không cần board, hãy giữ lại một tệp ONNX ở cùng
hình dạng cố định và so sánh nó trong trình mô phỏng, như ở trên.

## Ràng buộc

Có bốn tổ hợp biên dịch được, và chúng là các biến thể mô hình chứ không phải các
họ mô hình:

| Biến thể | Tác vụ | Canvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Mọi thứ khác đều bị từ chối trước khi biên dịch, kèm thông báo rằng RKNN trong
phiên bản này chỉ giới hạn ở đúng các biến thể phát hiện đối tượng đã được kiểm
thử trên trình mô phỏng. Có kết quả biên dịch được cho những mô hình khác, nhưng
chúng cố ý không được trình bày như là đã hỗ trợ: trong cùng lần đo đó, RF-DETR để
lại hai node `GridSample` của decoder chưa được hạ xuống, còn D-FINE, RT-DETR,
RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 và EC thì biên dịch và mô phỏng được nhưng cho
ra các đầu ra đã giải mã sai lệch đáng kể.

Batch 1, hình dạng tĩnh, opset 19. `half=True` bị từ chối, vì RKNN không phơi ra
hợp đồng `half` của LibreYOLO, và `int8=True` bị từ chối cho đến khi có dữ liệu
hiệu chuẩn đại diện và kết quả độ chính xác theo tác vụ.

Các target Rockchip khác đều bị từ chối: `rk3588` là nền tảng duy nhất đã được
kiểm chứng.

Để xem toàn bộ bảng họ mô hình và tác vụ, xem
[ma trận xuất mô hình](/docs/reference/export-matrix). Với một tổ hợp cụ thể:

<code-tabs name="support" />
