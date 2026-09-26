---
title: Python API
seo_title: Tham chiếu Python API LibreYOLO
description: >-
  Các tên LibreYOLO export ở cấp package: năm factory, lớp họ, payload Results,
  backend, validator, tracker và helper dữ liệu.
lead: >-
  Giao diện Python công khai của LibreYOLO là danh sách __all__ trong
  libreyolo/__init__.py. Mọi thành phần trên trang này có thể import bằng from
  libreyolo import <name>; thành phần không nằm trong danh sách là nội bộ.
keywords:
  - libreyolo python api
  - import libreyolo
  - LibreYOLO factory
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Tên và signature được đọc từ libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py và
  libreyolo/ensemble/model.py ở v1.5.0.
snippets:
  usage:
    - label: Nạp mọi thứ qua một factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Source là một ảnh trả về một Results; danh sách hoặc thư mục
        # trả về danh sách Results.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Import trực tiếp lớp họ
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Năm điểm vào
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Factory dò trọng số trên các họ không có prompt.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Hai detector trở lên phía sau một giao diện dự đoán.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Ba factory còn lại cần cài gói bổ sung:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Điểm vào

Năm callable nạp mô hình. Chúng được tách theo call contract chứ không theo
kiến trúc.

| Factory | Nội dung nạp | Prompt lúc gọi | Gói bổ sung cần thiết |
|---|---|---|---|
| `LibreYOLO` | Các họ không có prompt, bằng cách dò checkpoint hoặc hậu tố file | | |
| `LibreSAM` | Segmenter có prompt, theo alias kích thước | Điểm, box hoặc văn bản khái niệm | `sam` |
| `LibreVLM` | Detector vision-language sinh, theo alias | Vocabulary lớp hoặc prompt tự do | `vlm` |
| `LibreOpenVocab` | Detector được điều kiện hóa bằng văn bản, theo alias | Vocabulary lớp | `openvocab` |
| `LibreEnsemble` | Hai detector trở lên được hợp nhất vào một giao diện | | |

<code-tabs name="factories" />

`LibreYOLO` là factory duy nhất đọc file. Ba factory còn lại nhận alias chuỗi và
phân giải sang repo Hugging Face, vì vậy đối số là tên mô hình chứ không phải
đường dẫn.

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path` nhận checkpoint `.pt`, file ONNX `.onnx`, ExecuTorch `.pte`, MNN
`.mnn`, TensorRT `.engine`, thư mục OpenVINO, Paddle hoặc ncnn, hay URL mô hình
Triton HTTP hoặc HTTPS. `size` và `nb_classes` được đọc từ checkpoint khi bỏ
qua. `compute_units` chỉ được đọc khi nạp `.mlpackage` CoreML và nhận một trong
`all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne`. `task` nhận mọi tên tác vụ chuẩn
từ `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Lớp họ

Mọi họ mà factory có thể trả về cũng được export theo tên, vì vậy có thể dựng
trực tiếp lớp khi biết trước checkpoint. Constructor tuân theo
`BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` không có giá trị mặc định trên lớp họ, đây là điểm khác factory. YOLO9
và các biến thể chèn `reg_max: int = 16` sau `size`.

Các họ detection và đa tác vụ: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Các họ dense prediction: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Các họ classification và embedding: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Các tác vụ khác: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Các cấp sibling cũng export lớp họ: `LibreSAM1`, `LibreSAM2`, `LibreSAM3`,
`LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`; `LibreGroundingDINO`,
`LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`, `LibreQwen3VL`,
`LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`,
`LibreLocateAnything`, `LibreMODUS` (cũng viết là `LibreModus`).

## Giao diện dự đoán

Gọi mô hình sẽ chạy inference. `predict` là alias của `__call__`, nên hai cách
có thể thay thế nhau.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

Source là một ảnh trả về một `Results`. Danh sách, tuple hoặc thư mục trả về
danh sách, còn `stream=True` trả về generator. Các phương thức khác trên object
mô hình được mô tả tại [trang model API](/docs/reference/model-api).

## Payload Results

`Results` và mười tám lớp payload được export ở cấp package: `Results`,
`Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Từng lớp được mô tả trong [các kiểu Results](/docs/reference/results-types).

## Backend

Artifact đã xuất được nạp qua `LibreYOLO()` theo hậu tố file nên hiếm khi cần
dựng thủ công lớp backend. Chúng được export cho trường hợp phải chọn tường
minh: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`,
`TritonBackend`, `NcnnBackend`, `CoreMLBackend`, cùng `create_triton_config`.
`BaseExporter` là registry exporter phía sau `model.export()`.

## Validator

`model.val()` dispatch đến validator đúng theo tác vụ, nên các lớp sau được
export để dùng trực tiếp và tạo subclass: `DetectionValidator`,
`SegmentationValidator`, `PoseValidator`, `SemanticValidator`,
`PanopticValidator`, `DepthValidator`, `NormalValidator`, `EdgeValidator`, và
`ValidationConfig` dùng chung.

## Tracking

`model.track()` chọn tracker theo tên. Lớp tracker và dataclass cấu hình cũng
được export: `ByteTracker` với `TrackConfig`, `BoTSortTracker` với
`BoTSortConfig`, và `OCSortTracker` với `OCSortConfig`.

## Helper dữ liệu

`DATASETS_DIR` là thư mục gốc dataset đã phân giải, `load_data_config` đọc YAML
dataset và `check_dataset` xác thực file. Các loader theo tác vụ được nêu trong
[Định dạng dataset](/docs/reference/dataset-formats) nằm trong `libreyolo.data`
thay vì cấp package.

## Gallery và distillation

`Gallery` và `FaceGallery` giữ vector danh tính đã đăng ký cho tác vụ `embed`
và tạo payload `Identities`. `Distiller` cùng `get_distill_config` điều khiển
huấn luyện teacher-student.

## Asset

`SAMPLE_IMAGE` là đường dẫn tuyệt đối đến ảnh đi kèm package, nên mọi snippet
trong tài liệu chạy được mà không cần tải ảnh trước.

## Lazy import và lớp đổi tên

Phần lớn tên cấp sibling, backend, validator và helper dữ liệu phân giải qua
`__getattr__` cấp module, nên import `libreyolo` không import dependency của
chúng. Import vẫn thất bại với thông báo rõ ràng khi thiếu gói bổ sung cần thiết.

Hai tên lớp đã được đổi và cách viết cũ vẫn phân giải kèm `DeprecationWarning`:
`LibreYOLORTDETR` giờ là `LibreRTDETR`, còn `LibreYOLORFDETR` giờ là
`LibreRFDETR`.
