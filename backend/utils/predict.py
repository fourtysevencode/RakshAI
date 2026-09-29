from pathlib import Path
import ast
import logging
import threading
import numpy as np
import cv2

logger = logging.getLogger("RakshAI.Predict")

_MODEL = None
_MODEL_LOADED = False
_MODEL_LOAD_ERROR = None
# Class id -> name mapping, read from the model itself (YOLO export metadata)
CLASS_NAMES: dict = {}
# Ultralytics predictors / ORT sessions are shared across background tasks
_PREDICT_LOCK = threading.Lock()

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Determine repo root so that we look for models at the repo root/models/best.onnx
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best.onnx"


def model_info() -> dict:
    """Model status for the health endpoint and the report's methodology section."""
    _load_model()
    backend = None
    if isinstance(_MODEL, tuple):
        backend = "onnxruntime"
    elif _MODEL is not None:
        backend = "ultralytics"
    return {
        "loaded": _MODEL is not None,
        "backend": backend,
        "path": MODEL_PATH.name,
        "classes": [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)],
        "error": str(_MODEL_LOAD_ERROR) if _MODEL is None and _MODEL_LOAD_ERROR else None,
    }


def _load_model():
    global _MODEL, _MODEL_LOADED, _MODEL_LOAD_ERROR, CLASS_NAMES
    if _MODEL_LOADED:
        return
    _MODEL_LOADED = True
    if not MODEL_PATH.exists():
        _MODEL_LOAD_ERROR = FileNotFoundError(str(MODEL_PATH))
        logger.error(f"Model not found at {MODEL_PATH}")
        return
    try:
        # Try Ultralytics YOLO with ONNX model path first
        from ultralytics import YOLO  # type: ignore
        _MODEL = YOLO(str(MODEL_PATH), task="detect")
        CLASS_NAMES = dict(_MODEL.names) if _MODEL.names else {}
        return
    except Exception as e:
        _MODEL_LOAD_ERROR = e
        _MODEL = None

    # If Ultralytics path failed or isn't available, try a plain ONNXRuntime path
    try:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(str(MODEL_PATH))
        names = sess.get_modelmeta().custom_metadata_map.get("names")
        CLASS_NAMES = ast.literal_eval(names) if names else {}
        _MODEL = ("onnxruntime", sess)  # type: ignore
    except Exception as e:
        _MODEL_LOAD_ERROR = e
        _MODEL = None
        logger.error(f"Failed to load model {MODEL_PATH}: {e}")


def _letterbox(frame: np.ndarray, size: int = 640):
    """Resize keeping aspect ratio and pad to size x size (YOLO preprocessing)."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _parse_onnx_preds(pred: np.ndarray, scale: float, pad_x: int, pad_y: int, orig_shape,
                      conf: float = CONF_THRESHOLD) -> list:
    """Parse raw YOLOv8/11 output of shape (1, 4 + num_classes, num_anchors).

    Rows are [cx, cy, w, h, cls_0 .. cls_n] in letterboxed pixel space; there is
    no objectness score and NMS has not been applied.
    """
    pred = np.squeeze(pred, axis=0).T  # (num_anchors, 4 + num_classes)
    boxes, scores = pred[:, :4], pred[:, 4:]
    cls_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), cls_ids]
    keep = confs > conf
    boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]
    if len(boxes) == 0:
        return []

    # cxcywh (letterboxed) -> xyxy (original frame)
    oh, ow = orig_shape[:2]
    x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale, 0, ow)
    y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale, 0, oh)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale, 0, ow)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale, 0, oh)

    detections = []
    # Class-aware NMS: offset boxes per class so different classes don't suppress each other
    for cid in np.unique(cls_ids):
        idx = np.where(cls_ids == cid)[0]
        rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in idx]
        kept = cv2.dnn.NMSBoxes(rects, confs[idx].tolist(), conf, IOU_THRESHOLD)
        for k in np.array(kept).flatten():
            i = idx[int(k)]
            detections.append({
                "class_id": int(cid),
                "class_name": CLASS_NAMES.get(int(cid), str(int(cid))),
                "confidence": float(confs[i]),
                "bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
            })
    return detections


def predict_frame(frame: np.ndarray, conf: float = CONF_THRESHOLD) -> list:
    """Predict detections for a single BGR frame (as read by OpenCV).

    `conf` is the minimum confidence kept; tracking passes a lower value so
    ByteTrack can use weak person boxes in its second association pass.

    Returns a list of detections, where each detection is a dict with keys:
    - class_id
    - class_name
    - confidence
    - bbox [x1, y1, x2, y2]

    Raises RuntimeError if the model could not be loaded, so callers don't
    silently treat "no model" as "no detections".
    """
    _load_model()
    if _MODEL is None:
        raise RuntimeError(f"Detection model unavailable: {_MODEL_LOAD_ERROR}")

    with _PREDICT_LOCK:
        if isinstance(_MODEL, tuple) and _MODEL[0] == "onnxruntime":
            sess = _MODEL[1]
            input_name = sess.get_inputs()[0].name
            img, scale, pad_x, pad_y = _letterbox(frame, 640)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = img.transpose(2, 0, 1)[np.newaxis, ...]
            outputs = sess.run(None, {input_name: input_tensor})
            return _parse_onnx_preds(outputs[0], scale, pad_x, pad_y, frame.shape, conf)

        # Ultralytics path: model(frame) returns a list of Results (one per image)
        results = _MODEL(frame, conf=conf, verbose=False)
    detections = []
    for res in results:
        names = res.names or CLASS_NAMES
        for xyxy, score, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            cls_id = int(cls)
            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": float(score),
                "bbox": [int(v) for v in xyxy],
            })
    return detections
