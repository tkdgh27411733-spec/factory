from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


MODEL_FILENAME = "lesson06_vision_model.joblib"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@st.cache_resource
def load_model_bundle(model_path: str) -> dict[str, Any]:
    """모델 파일을 한 번만 불러온다."""
    bundle = joblib.load(model_path)
    required = {
        "model",
        "feature_size",
        "operating_threshold",
        "quality_limits",
        "class_names",
        "feature_extractor",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(
            "모델 파일에 필요한 항목이 없습니다: "
            + ", ".join(sorted(missing))
        )
    if bundle["feature_extractor"] != "lesson06_hog_intensity_v1":
        raise ValueError(
            "앱과 호환되지 않는 특징 추출기입니다: "
            f"{bundle['feature_extractor']}"
        )
    return bundle


def quality_metrics(
    image: Image.Image,
    feature_size: tuple[int, int],
) -> dict[str, float]:
    """학습 코드와 동일한 방식으로 입력 이미지 품질을 계산한다."""
    array = np.asarray(image.resize(feature_size), dtype=np.float32)
    gx = np.diff(array, axis=1, prepend=array[:, :1])
    gy = np.diff(array, axis=0, prepend=array[:1, :])
    laplacian = (
        -4 * array
        + np.roll(array, 1, axis=0)
        + np.roll(array, -1, axis=0)
        + np.roll(array, 1, axis=1)
        + np.roll(array, -1, axis=1)
    )
    return {
        "brightness": float(array.mean()),
        "contrast": float(array.std()),
        "sharpness": float(laplacian.var()),
        "mean_gradient": float(np.hypot(gx, gy).mean()),
    }


def quality_failures(
    metrics: dict[str, float],
    limits: dict[str, float],
) -> list[str]:
    failures = []
    if metrics["brightness"] < limits["brightness_low"]:
        failures.append("밝기가 학습 범위보다 낮습니다.")
    if metrics["brightness"] > limits["brightness_high"]:
        failures.append("밝기가 학습 범위보다 높습니다.")
    if metrics["contrast"] < limits["contrast_low"]:
        failures.append("대비가 부족합니다.")
    if metrics["sharpness"] < limits["sharpness_low"]:
        failures.append("초점 또는 해상도가 부족합니다.")
    return failures


def extract_features(
    image: Image.Image,
    feature_size: tuple[int, int],
) -> np.ndarray:
    """6차시 모델 학습 때 사용한 HOG·밝기 특징을 추출한다."""
    array = np.asarray(
        image.resize(feature_size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    normalized = array / 255.0
    gx = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    gy = np.diff(normalized, axis=0, prepend=normalized[:1, :])
    magnitude = np.hypot(gx, gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    hog: list[float] = []
    bins = np.linspace(0, 180, 10)
    for row in range(0, feature_size[1], 8):
        for column in range(0, feature_size[0], 8):
            cell_angle = orientation[row : row + 8, column : column + 8]
            cell_weight = magnitude[row : row + 8, column : column + 8]
            histogram, _ = np.histogram(
                cell_angle,
                bins=bins,
                weights=cell_weight,
            )
            histogram = histogram / (histogram.sum() + 1e-6)
            hog.extend(histogram.tolist())

    intensity_histogram, _ = np.histogram(
        normalized,
        bins=16,
        range=(0, 1),
        density=True,
    )
    percentiles = np.percentile(
        normalized,
        [1, 5, 25, 50, 75, 95, 99],
    )
    extra = np.array(
        [
            normalized.mean(),
            normalized.std(),
            magnitude.mean(),
            np.percentile(magnitude, 90),
            np.percentile(magnitude, 99),
        ]
    )
    return np.concatenate(
        [
            np.asarray(hog),
            intensity_histogram,
            percentiles,
            extra,
        ]
    )


def render_app() -> None:
    st.set_page_config(
        page_title="비전검사 AI 체험",
        page_icon="🔍",
        layout="wide",
    )
    st.title("비전검사 AI 체험")
    st.caption(
        "이미지 한 장을 업로드하면 촬영 품질을 확인하고 "
        "정상 또는 불량 검토 후보로 분류합니다."
    )

    if not MODEL_PATH.is_file():
        st.error(f"모델 파일을 찾지 못했습니다: {MODEL_FILENAME}")
        st.info(
            "`app.py`와 `lesson06_vision_model.joblib`을 "
            "GitHub 저장소의 같은 폴더에 올려 주세요."
        )
        st.stop()

    try:
        bundle = load_model_bundle(str(MODEL_PATH))
    except Exception as error:
        st.error("저장된 모델을 불러오지 못했습니다.")
        st.code(str(error))
        st.stop()

    with st.sidebar:
        st.subheader("모델 정보")
        st.write(f"모델 파일: `{MODEL_FILENAME}`")
        st.write(f"학습 데이터: {bundle.get('dataset', 'KSDD')}")
        st.write(
            "판정 임계값: "
            f"{float(bundle['operating_threshold']) * 100:.1f}%"
        )
        st.warning(
            "학습에 사용한 제품과 촬영 조건이 다른 이미지는 "
            "정확한 판정이 어려울 수 있습니다."
        )

    uploaded = st.file_uploader(
        "검사할 JPG 또는 PNG 이미지를 선택하세요.",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("이미지를 업로드하면 분석 결과가 표시됩니다.")
        return
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("이미지는 10MB 이하만 업로드할 수 있습니다.")
        return

    try:
        image = Image.open(uploaded)
        image.verify()
        uploaded.seek(0)
        image = Image.open(uploaded).convert("L")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("지원되는 정상적인 이미지 파일이 아닙니다.")
        return

    feature_size = tuple(int(value) for value in bundle["feature_size"])
    metrics = quality_metrics(image, feature_size)
    failures = quality_failures(metrics, bundle["quality_limits"])

    image_column, result_column = st.columns([1.15, 1])
    with image_column:
        st.image(image, caption="업로드 이미지", use_container_width=True)

    with result_column:
        st.subheader("이미지 품질")
        columns = st.columns(3)
        columns[0].metric("밝기", f"{metrics['brightness']:.1f}")
        columns[1].metric("대비", f"{metrics['contrast']:.1f}")
        columns[2].metric("선명도", f"{metrics['sharpness']:.1f}")

        if failures:
            st.error("촬영 조건 부적합 — 재촬영 또는 사람 검토")
            for failure in failures:
                st.write(f"- {failure}")
            st.info("품질 기준을 통과하지 못해 모델 판정을 생략합니다.")
            return

        st.success("촬영 품질 기준 통과")
        feature = extract_features(image, feature_size).reshape(1, -1)
        expected_features = getattr(
            bundle["model"],
            "n_features_in_",
            feature.shape[1],
        )
        if feature.shape[1] != expected_features:
            st.error(
                "이미지 특징 수와 저장 모델의 입력 수가 일치하지 않습니다. "
                f"앱 {feature.shape[1]}개 / 모델 {expected_features}개"
            )
            return

        probability = float(bundle["model"].predict_proba(feature)[0, 1])
        threshold = float(bundle["operating_threshold"])

        st.subheader("모델 판정")
        st.metric("불량 점수", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"불량 검토 임계값: {threshold * 100:.1f}%")

        if probability >= threshold:
            st.error("불량 검토 후보")
        else:
            st.success("정상 후보")

    st.warning(
        "교육용 모델의 결과입니다. 실제 제품의 폐기나 공정 정지는 "
        "승인된 검사 기준과 작업자 확인을 거쳐 결정해야 합니다."
    )


if __name__ == "__main__":
    render_app()
