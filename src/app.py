"""Provide a Streamlit UI for training and viewing model results.

English:
    Lets users choose a local photo folder, prepare the dataset, run training,
    upload photos for prediction, view Grad-CAM, and inspect saved results.

한국어:
    사용자가 로컬 사진 폴더를 선택하여 데이터셋을 준비하고 학습을 실행하며,
    사진 예측, Grad-CAM, 저장된 결과 확인까지 웹 브라우저에서 할 수 있게 합니다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    HISTORY_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SUPPORTED_EXTENSIONS,
    get_device,
    load_checkpoint,
)
from gradcam import create_gradcam_visuals
from predict import predict_image


st.set_page_config(
    page_title="Photo Geolocation Prediction",
    layout="wide",
)


@st.cache_resource
def load_ui_model(model_path: str):
    """Load and cache the trained model across Streamlit reruns."""
    device = get_device()
    model, class_names, checkpoint = load_checkpoint(Path(model_path), device)
    return model, class_names, checkpoint, device


def format_location(location: str) -> str:
    """Convert folder-style class names into readable labels."""
    return location.replace("_", " ")


def count_raw_images(raw_dir: Path) -> dict[str, int]:
    """Count supported image files in each class folder."""
    if not raw_dir.exists():
        return {}
    counts = {}
    for class_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return counts


def run_command(command: list[str], log_label: str) -> bool:
    """Run a project command and stream logs into the Streamlit page."""
    st.markdown(f"#### {log_label}")
    log_box = st.empty()
    lines: list[str] = []
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-80:]))
    return_code = process.wait()
    if return_code == 0:
        st.success(f"완료: {log_label}")
        return True
    st.error(f"실패: {log_label} (exit code {return_code})")
    return False


def show_data_and_training_ui() -> None:
    """Render local folder selection, dataset preparation, and training controls."""
    st.subheader("데이터 준비 및 학습 실행 / Prepare Data and Train")
    st.write(
        "Google Takeout으로 받은 사진을 위치별 하위 폴더로 정리한 뒤, "
        "그 상위 폴더 경로를 입력하세요. 예: `data/raw/`"
    )

    default_raw_dir = str(RAW_DATA_DIR)
    raw_dir_text = st.text_input(
        "사진 폴더 경로 / Photo folder path",
        value=default_raw_dir,
        help="이 폴더 안에 Santa_Cruz, San_Francisco 같은 클래스 폴더가 있어야 합니다.",
    )
    raw_dir = Path(raw_dir_text).expanduser()
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    st.caption(f"Resolved path: `{raw_dir}`")
    counts = count_raw_images(raw_dir)
    if counts:
        count_frame = pd.DataFrame(
            [{"location": name, "image_count": count} for name, count in counts.items()]
        )
        st.dataframe(count_frame, hide_index=True, use_container_width=True)
        if any(count < 7 for count in counts.values()):
            st.warning(
                "각 위치 클래스에는 최소 7장의 유효 이미지가 필요합니다. "
                "실제로는 클래스마다 더 많은 사진을 권장합니다."
            )
    else:
        st.info("아직 읽을 수 있는 위치별 이미지 폴더를 찾지 못했습니다.")

    st.markdown("#### 학습 설정 / Training Settings")
    setting_columns = st.columns(4)
    batch_size = setting_columns[0].number_input(
        "Batch size", min_value=1, max_value=256, value=16, step=1
    )
    head_epochs = setting_columns[1].number_input(
        "Head epochs", min_value=1, max_value=100, value=5, step=1
    )
    fine_tune_epochs = setting_columns[2].number_input(
        "Fine-tune epochs", min_value=0, max_value=100, value=3, step=1
    )
    num_workers = setting_columns[3].number_input(
        "Workers", min_value=0, max_value=8, value=0, step=1
    )
    learning_rate = st.number_input(
        "Classifier learning rate",
        min_value=0.000001,
        max_value=1.0,
        value=0.001,
        step=0.0001,
        format="%.6f",
    )
    fine_tune_learning_rate = st.number_input(
        "Fine-tune learning rate",
        min_value=0.000001,
        max_value=1.0,
        value=0.0001,
        step=0.00001,
        format="%.6f",
    )
    run_evaluation = st.checkbox(
        "학습 후 테스트 평가까지 실행 / Run evaluation after training",
        value=True,
    )

    st.markdown("#### 실행 / Run")
    prepare_only = st.button("데이터셋만 준비 / Prepare Dataset Only")
    train_button = st.button("데이터 준비 후 학습 시작 / Prepare and Train")

    if prepare_only or train_button:
        if not raw_dir.exists():
            st.error(f"사진 폴더가 존재하지 않습니다: {raw_dir}")
            return
        if len(counts) < 2:
            st.error("최소 두 개 이상의 위치 클래스 폴더가 필요합니다.")
            return

        prepare_command = [
            sys.executable,
            "src/prepare_dataset.py",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(PROCESSED_DATA_DIR),
            "--overwrite",
        ]
        if not run_command(prepare_command, "1. Dataset preparation"):
            return
        if prepare_only:
            return

        train_command = [
            sys.executable,
            "src/train.py",
            "--data-dir",
            str(PROCESSED_DATA_DIR),
            "--batch-size",
            str(int(batch_size)),
            "--head-epochs",
            str(int(head_epochs)),
            "--fine-tune-epochs",
            str(int(fine_tune_epochs)),
            "--learning-rate",
            str(float(learning_rate)),
            "--fine-tune-learning-rate",
            str(float(fine_tune_learning_rate)),
            "--num-workers",
            str(int(num_workers)),
        ]
        if not run_command(train_command, "2. Model training"):
            return

        if run_evaluation:
            evaluate_command = [
                sys.executable,
                "src/evaluate.py",
                "--data-dir",
                str(PROCESSED_DATA_DIR),
            ]
            if not run_command(evaluate_command, "3. Test evaluation"):
                return

        load_ui_model.clear()
        st.success("전체 학습 작업이 완료되었습니다. 예측 탭과 결과 탭에서 확인하세요.")


def show_prediction_ui() -> None:
    """Render image upload, prediction, probability, and Grad-CAM controls."""
    st.subheader("새 사진 위치 예측 / Predict a New Photo")
    st.write(
        "사진을 업로드하면 학습된 ResNet50 모델이 위치 후보와 확률을 보여줍니다. "
        "EXIF GPS 정보는 사용하지 않습니다."
    )

    if not BEST_MODEL_PATH.exists():
        st.info("예측하려면 먼저 데이터/학습 탭에서 모델을 학습하세요.")
        return

    uploaded_file = st.file_uploader(
        "JPG, PNG, HEIC 이미지를 선택하세요.",
        type=["jpg", "jpeg", "png", "heic", "heif"],
    )
    if uploaded_file is None:
        st.info("예측할 사진을 업로드하세요.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as error:
        st.error(f"이미지를 읽을 수 없습니다: {error}")
        return

    with st.spinner("모델을 불러오고 위치를 예측하는 중입니다..."):
        model, class_names, checkpoint, device = load_ui_model(str(BEST_MODEL_PATH))
        predictions, probabilities = predict_image(
            model, image, class_names, device, top_k=3
        )

    top_prediction = predictions[0]
    image_column, result_column = st.columns([1.2, 1])
    with image_column:
        st.image(image, caption="업로드한 이미지 / Uploaded Image", use_container_width=True)
    with result_column:
        st.metric(
            "가장 가능성 높은 위치 / Top Prediction",
            format_location(str(top_prediction["location"])),
            f"{float(top_prediction['probability']):.2%}",
        )
        st.caption(
            f"실행 장치: {device} | 체크포인트 epoch: {checkpoint['epoch']} | "
            f"검증 정확도: {checkpoint['val_accuracy']:.2%}"
        )
        prediction_frame = pd.DataFrame(predictions)
        prediction_frame["location"] = prediction_frame["location"].map(format_location)
        prediction_frame["probability_percent"] = (
            prediction_frame["probability"].astype(float) * 100
        )
        st.dataframe(
            prediction_frame[["location", "probability_percent"]].rename(
                columns={
                    "location": "Location",
                    "probability_percent": "Probability (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(
            prediction_frame.set_index("location")["probability_percent"]
        )

    if st.checkbox("Grad-CAM으로 모델이 본 영역 표시 / Show Grad-CAM", value=True):
        class_index = int(probabilities.argmax().item())
        with st.spinner("Grad-CAM을 생성하는 중입니다..."):
            heatmap, overlay = create_gradcam_visuals(
                model, image, device, class_index
            )
        original_column, heatmap_column, overlay_column = st.columns(3)
        original_column.image(image, caption="Original", use_container_width=True)
        heatmap_column.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            clamp=True,
            use_container_width=True,
        )
        overlay_column.image(
            overlay,
            caption="Prediction Focus",
            use_container_width=True,
        )


def show_training_results_ui() -> None:
    """Render saved learning curves and evaluation artifacts."""
    st.subheader("학습 및 평가 결과 / Training and Evaluation Results")
    st.write("`train.py`와 `evaluate.py`가 저장한 결과를 한 화면에서 확인합니다.")

    history_path = HISTORY_PATH
    report_path = REPORTS_DIR / "classification_report.csv"
    summary_path = REPORTS_DIR / "evaluation_summary.txt"
    confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"

    if history_path.exists():
        history = pd.read_csv(history_path)
        st.markdown("#### 학습 곡선 / Learning Curves")
        loss_column, accuracy_column = st.columns(2)
        loss_column.line_chart(
            history.set_index("epoch")[["train_loss", "val_loss"]],
            y_label="Loss",
        )
        accuracy_column.line_chart(
            history.set_index("epoch")[["train_accuracy", "val_accuracy"]],
            y_label="Accuracy",
        )
        with st.expander("학습 기록 CSV 보기 / View Training History"):
            st.dataframe(history, hide_index=True, use_container_width=True)
    else:
        st.info("학습 기록이 없습니다. 먼저 `python src/train.py`를 실행하세요.")

    if report_path.exists():
        st.markdown("#### 클래스별 성능 / Per-Class Metrics")
        report = pd.read_csv(report_path, index_col=0)
        st.dataframe(report, use_container_width=True)

    if confusion_matrix_path.exists():
        st.markdown("#### 혼동 행렬 / Confusion Matrix")
        st.image(str(confusion_matrix_path), use_container_width=True)
    elif history_path.exists():
        st.info("혼동 행렬이 없습니다. `python src/evaluate.py`를 실행하세요.")

    if summary_path.exists():
        with st.expander("평가 요약 보기 / View Evaluation Summary"):
            st.text(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    st.title("Photo Geolocation Prediction using Deep Learning")
    st.caption(
        "ResNet50 transfer learning training and result viewer | "
        "ResNet50 전이 학습 실행 및 결과 확인 UI"
    )

    training_tab, prediction_tab, results_tab = st.tabs(
        [
            "데이터/학습 / Data & Training",
            "사진 예측 / Photo Prediction",
            "학습 결과 / Training Results",
        ]
    )
    with training_tab:
        show_data_and_training_ui()
    with prediction_tab:
        show_prediction_ui()
    with results_tab:
        show_training_results_ui()


if __name__ == "__main__":
    main()
