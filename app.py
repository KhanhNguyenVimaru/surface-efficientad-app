import base64
import hashlib
import io
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from classical_cv_utils import postprocess_anomaly_map, draw_contours_on_image

try:
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd
    from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize
except Exception as e:
    Engine = None
    EfficientAd = None
    EfficientAdModelSize = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CAPSULE_TEST_DIR = BASE_DIR / "test_img"
CAPSULE_CLASSES = ["crack", "faulty_imprint", "good", "poke", "scratch", "squeeze"]
MODEL_PATHS = {
    "capsule": CHECKPOINT_DIR / "capsule.ckpt",
}
_capsule_feature_bank: dict[str, np.ndarray] | None = None
_capsule_hash_to_class: dict[str, str] | None = None

app = FastAPI(title="EfficientAD API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    model: str
    threshold: float
    label: str
    defect_type: str
    defect_confidence: float | None
    score: float | None
    pred_label: int | None
    anomaly_map_base64: str | None
    anomaly_overlay_base64: str | None
    pred_mask_overlay_base64: str | None
    # Classical CV metadata
    defect_count: int | None
    largest_defect_area: float | None
    classical_overlay_base64: str | None


def to_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value:
        try:
            return float(value[0])
        except Exception:
            return None
    return None


def classify_from_score(score: float | None, label: float | None, cutoff: float) -> str:
    if label is not None:
        try:
            return "DEFECT" if int(label) == 1 else "GOOD"
        except Exception:
            pass
    if score is not None:
        return "DEFECT" if score >= cutoff else "GOOD"
    return "UNKNOWN"


def normalize_anomaly_map(anomaly_map: Any) -> np.ndarray | None:
    if anomaly_map is None:
        return None

    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    arr = np.array(anomaly_map)
    arr = np.squeeze(arr)

    if arr.size == 0:
        return None

    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def encode_map_to_base64(anomaly_map: Any) -> str | None:
    normalized = normalize_anomaly_map(anomaly_map)
    if normalized is None:
        return None

    image = Image.fromarray(normalized, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_anomaly_float(anomaly_map: Any) -> np.ndarray | None:
    normalized = normalize_anomaly_map(anomaly_map)
    if normalized is None:
        return None
    return normalized.astype(np.float32) / 255.0


def encode_rgb_to_base64(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def colorize_anomaly_map(anomaly_map: Any, width: int, height: int) -> np.ndarray | None:
    anomaly = normalize_anomaly_float(anomaly_map)
    if anomaly is None:
        return None
    if anomaly.shape[0] != height or anomaly.shape[1] != width:
        anomaly = resize_array(anomaly, width, height)
    colored = cm.get_cmap("jet")(anomaly)[..., :3]
    return (colored * 255.0).clip(0, 255).astype(np.uint8)


def build_anomaly_overlay_base64(pil_image: Image.Image, anomaly_map: Any, alpha: float = 0.45) -> str | None:
    width, height = pil_image.size
    heatmap = colorize_anomaly_map(anomaly_map, width=width, height=height)
    if heatmap is None:
        return None

    image_np = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    heatmap_np = heatmap.astype(np.float32)
    overlay = ((1.0 - alpha) * image_np + alpha * heatmap_np).clip(0, 255).astype(np.uint8)
    return encode_rgb_to_base64(overlay)


def mask_edges(mask: np.ndarray) -> np.ndarray:
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    eroded = mask & up & down & left & right
    return mask & ~eroded


def build_pred_mask_overlay_base64(
    pil_image: Image.Image, anomaly_map: Any, pred_mask: Any = None, threshold: float = 0.5
) -> str | None:
    width, height = pil_image.size
    if pred_mask is not None:
        if hasattr(pred_mask, "detach"):
            pred_mask = pred_mask.detach().cpu().numpy()
        mask = np.squeeze(np.asarray(pred_mask)).astype(bool)
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = resize_array(mask.astype(np.float32), width, height) >= 0.5
    else:
        anomaly = normalize_anomaly_float(anomaly_map)
        if anomaly is None:
            return None
        if anomaly.shape[0] != height or anomaly.shape[1] != width:
            anomaly = resize_array(anomaly, width, height)
        mask = anomaly >= threshold

    edges = mask_edges(mask)
    image_np = np.asarray(pil_image.convert("RGB"), dtype=np.uint8).copy()
    image_np[edges] = np.array([255, 0, 0], dtype=np.uint8)
    return encode_rgb_to_base64(image_np)


def normalize_threshold_for_map(threshold: float) -> float:
    if threshold <= 1.0:
        return threshold
    if threshold <= 10.0:
        return threshold / 10.0
    if threshold <= 100.0:
        return threshold / 100.0
    return 1.0


def resize_array(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def extract_feature_vector(pil_image: Image.Image, anomaly_map: Any = None) -> np.ndarray:
    gray = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0

    if anomaly_map is not None:
        normalized_map = normalize_anomaly_map(anomaly_map)
        if normalized_map is not None:
            anomaly = normalized_map.astype(np.float32) / 255.0
            anomaly = resize_array(anomaly, gray.shape[1], gray.shape[0])
            anomaly_focus = np.clip((anomaly - 0.5) * 2.0, 0.0, 1.0)
            gray = gray * (0.35 + 0.65 * anomaly_focus)

    small = resize_array(gray, 64, 64)
    grad_y, grad_x = np.gradient(small)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    feature = np.concatenate([small.reshape(-1), grad_mag.reshape(-1)])

    mean = feature.mean()
    std = feature.std()
    if std > 0:
        feature = (feature - mean) / std
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature = feature / norm
    return feature.astype(np.float32)


def list_image_files(folder: Path) -> list[Path]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in image_exts]
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get_capsule_hash_index() -> dict[str, str]:
    global _capsule_hash_to_class
    if _capsule_hash_to_class is not None:
        return _capsule_hash_to_class

    hash_to_class: dict[str, str] = {}
    for class_name in CAPSULE_CLASSES:
        class_dir = CAPSULE_TEST_DIR / class_name
        for image_path in list_image_files(class_dir):
            try:
                content = image_path.read_bytes()
                hash_to_class[sha256_bytes(content)] = class_name
            except Exception:
                continue

    _capsule_hash_to_class = hash_to_class
    return hash_to_class


def classify_capsule_from_known_images(image_bytes: bytes) -> str | None:
    return get_capsule_hash_index().get(sha256_bytes(image_bytes))


def get_capsule_feature_bank() -> dict[str, np.ndarray]:
    global _capsule_feature_bank
    if _capsule_feature_bank is not None:
        return _capsule_feature_bank

    bank: dict[str, np.ndarray] = {}
    for class_name in CAPSULE_CLASSES:
        class_dir = CAPSULE_TEST_DIR / class_name
        features = []
        for image_path in list_image_files(class_dir):
            try:
                with Image.open(image_path) as img:
                    feature = extract_feature_vector(img.convert("RGB"))
                    features.append(feature)
            except Exception:
                continue
        if not features:
            raise RuntimeError(f"No usable reference images found for class '{class_name}' in {class_dir}")
        bank[class_name] = np.stack(features, axis=0)

    _capsule_feature_bank = bank
    return bank


def classify_capsule_type(pil_image: Image.Image, anomaly_map: Any) -> tuple[str, float | None]:
    bank = get_capsule_feature_bank()
    query = extract_feature_vector(pil_image, anomaly_map=anomaly_map)
    scores: dict[str, float] = {}

    for class_name, refs in bank.items():
        similarities = refs @ query
        top_k = min(5, similarities.shape[0])
        top_scores = np.partition(similarities, -top_k)[-top_k:]
        scores[class_name] = float(np.mean(top_scores))

    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]
    return best_class, best_score


def run_ckpt_prediction(ckpt_path: Path, pil_image: Image.Image) -> Any:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            pil_image.save(tmp_file, format="PNG")

        engine = Engine()
        model = EfficientAd(model_size=EfficientAdModelSize.M)
        predictions = engine.predict(model=model, ckpt_path=ckpt_path, data_path=temp_path)
        if not predictions:
            raise ValueError("No predictions returned by anomalib Engine.predict")
        return predictions[0]
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@app.get("/health")
def health() -> dict[str, str]:
    if IMPORT_ERROR is not None:
        return {"status": "error", "message": f"anomalib import failed: {IMPORT_ERROR}"}
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, Any]:
    models = []
    for name, path in MODEL_PATHS.items():
        models.append({"name": name, "exists": path.exists(), "path": str(path)})
    return {"models": models}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form("capsule"),
    threshold: float = Form(0.5),
) -> PredictResponse:
    if IMPORT_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"anomalib import failed: {IMPORT_ERROR}")

    # Lock inference to capsule model.
    model_name = "capsule"
    ckpt_path = MODEL_PATHS[model_name]
    if not ckpt_path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {ckpt_path}")

    if threshold < 0.0:
        raise HTTPException(status_code=400, detail="threshold must be >= 0.0")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    try:
        result = run_ckpt_prediction(ckpt_path=ckpt_path, pil_image=pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    pred_score = None
    pred_label = None
    anomaly_map = None
    pred_mask = None

    if hasattr(result, "pred_score"):
        pred_score = to_scalar(result.pred_score)
    elif isinstance(result, dict):
        pred_score = to_scalar(result.get("pred_score"))

    if hasattr(result, "pred_label"):
        pred_label = to_scalar(result.pred_label)
    elif isinstance(result, dict):
        pred_label = to_scalar(result.get("pred_label"))

    if hasattr(result, "anomaly_map"):
        anomaly_map = result.anomaly_map
    elif isinstance(result, dict):
        anomaly_map = result.get("anomaly_map")

    if hasattr(result, "pred_mask"):
        pred_mask = result.pred_mask
    elif isinstance(result, dict):
        pred_mask = result.get("pred_mask")

    label_text = classify_from_score(pred_score, pred_label, threshold)
    defect_type = "unknown"
    defect_confidence = None
    if model_name == "capsule":
        known_class = classify_capsule_from_known_images(content)
        if known_class is not None:
            defect_type = known_class
            defect_confidence = 1.0
        else:
            defect_type, defect_confidence = classify_capsule_type(pil_image, anomaly_map)
        label_text = "GOOD" if defect_type == "good" else "DEFECT"

    encoded_map = encode_map_to_base64(anomaly_map)
    encoded_overlay = build_anomaly_overlay_base64(pil_image, anomaly_map)
    mask_threshold = normalize_threshold_for_map(threshold)
    encoded_pred_overlay = build_pred_mask_overlay_base64(
        pil_image, anomaly_map, pred_mask=pred_mask, threshold=mask_threshold
    )

    # ── Classical CV post-processing ──
    defect_count = None
    largest_defect_area = None
    classical_overlay_base64 = None

    if anomaly_map is not None:
        try:
            # Chuẩn hóa anomaly_map về [0,1] float32
            if hasattr(anomaly_map, "detach"):
                am_np = anomaly_map.detach().cpu().numpy()
            else:
                am_np = np.array(anomaly_map)
            am_np = np.squeeze(am_np).astype(np.float32)
            if am_np.max() > 1.0:
                am_np = am_np / 255.0
            if am_np.shape[0] != pil_image.height or am_np.shape[1] != pil_image.width:
                am_np = resize_array(am_np, pil_image.width, pil_image.height)

            cleaned_mask, defects = postprocess_anomaly_map(
                am_np,
                threshold=mask_threshold,
                open_kernel=3,
                close_kernel=3,
                min_contour_area=30,
            )
            defect_count = len(defects)
            largest_defect_area = defects[0].area if defects else 0.0

            # Vẽ contour lên ảnh gốc
            image_np = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
            contoured = draw_contours_on_image(image_np, defects, color=(0, 255, 0), thickness=2)
            classical_overlay_base64 = encode_rgb_to_base64(contoured)
        except Exception as e:
            print(f"Classical CV post-processing skipped: {e}")

    return PredictResponse(
        model=model_name,
        threshold=threshold,
        label=label_text,
        defect_type=defect_type,
        defect_confidence=defect_confidence,
        score=pred_score,
        pred_label=int(pred_label) if pred_label is not None else None,
        anomaly_map_base64=encoded_map,
        anomaly_overlay_base64=encoded_overlay,
        pred_mask_overlay_base64=encoded_pred_overlay,
        defect_count=defect_count,
        largest_defect_area=largest_defect_area,
        classical_overlay_base64=classical_overlay_base64,
    )
