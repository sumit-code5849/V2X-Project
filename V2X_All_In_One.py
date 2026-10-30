"""V2X Message Classification — consolidated single-file implementation.

Run the Streamlit interface:
    streamlit run V2X_All_In_One.py

Train all four architectures:
    python V2X_All_In_One.py train --data data/V2X_realistic_inspired_dataset_12000.csv --epochs 10 --out models_realistic

Evaluate a saved model:
    python V2X_All_In_One.py evaluate --data data/V2X_realistic_inspired_dataset_12000.csv --model models_realistic/CNN_BiRNN.keras

The data and trained .keras model files remain external assets. This file consolidates
Python source code; it intentionally does not embed large datasets or model binaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. CONFIGURATION (formerly src/config.py)
# -----------------------------------------------------------------------------

CLASSES = ["Emergency", "Hazard", "Roadwork", "Traffic", "Weather", "Spam"]
PRIORITIES = ["Low", "Medium", "High", "Unknown"]
SEQ_LEN = 40
EMBED_DIM = 64
RNN_UNITS = 64
DROPOUT = 0.25
SEED = 42
ARCHITECTURES = ["CNN-BiGRU", "CNN-BiLSTM", "CNN-StackLSTM", "CNN-BiRNN"]
REQUIRED_COLUMNS = ["Message Content", "Message Type", "Priority", "Spam"]

# -----------------------------------------------------------------------------
# 2. DATA LOADING AND PREPROCESSING (formerly src/data_utils.py)
# -----------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load, validate and clean a V2X CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).drop_duplicates().copy()
    df["Message Content"] = df["Message Content"].astype(str).str.strip()
    df = df[(df["Message Content"] != "") & df["Message Type"].isin(CLASSES)].copy()
    if df.empty:
        raise ValueError("No valid rows remain after dataset cleaning.")
    return df


def prepare(df: pd.DataFrame):
    """Create reproducible, message-level train/validation/test partitions (70/15/15)."""
    df = df.copy()
    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    df["type_id"] = df["Message Type"].map(class_to_id)

    spam = df["Spam"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    if spam.isna().any():
        bad_values = sorted(df.loc[spam.isna(), "Spam"].astype(str).unique().tolist())
        raise ValueError(f"Spam labels must be Yes/No or binary values. Invalid: {bad_values}")
    df["spam_id"] = spam.astype("int32")
    df["Priority"] = df["Priority"].where(df["Priority"].isin(PRIORITIES[:-1]), "Unknown")

    # Split unique message contents to prevent the same message appearing in
    # multiple partitions. Stratification preserves class proportions.
    unique_messages = df[["Message Content", "type_id"]].drop_duplicates("Message Content")
    train_messages, remaining_messages = train_test_split(
        unique_messages,
        test_size=0.30,
        random_state=SEED,
        stratify=unique_messages["type_id"],
    )
    val_messages, test_messages = train_test_split(
        remaining_messages,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_messages["type_id"],
    )

    train_df = df[df["Message Content"].isin(train_messages["Message Content"])].copy()
    val_df = df[df["Message Content"].isin(val_messages["Message Content"])].copy()
    test_df = df[df["Message Content"].isin(test_messages["Message Content"])].copy()
    return train_df, val_df, test_df


def xy(df: pd.DataFrame):
    """Convert a dataframe partition into Keras input and target dictionaries."""
    x = {
        "message_content": df["Message Content"].astype(str).values,
        "priority": df["Priority"].astype(str).values,
    }
    y = {
        "message_type": df["type_id"].to_numpy(dtype="int32"),
        "spam": df["spam_id"].to_numpy(dtype="float32"),
    }
    return x, y


def tensors(x: dict[str, Any]):
    """Convert string inputs to tensors; also accepts already-converted inputs."""
    return {
        "message_content": tf.convert_to_tensor(x["message_content"], dtype=tf.string),
        "priority": tf.convert_to_tensor(x["priority"], dtype=tf.string),
    }

# -----------------------------------------------------------------------------
# 3. MODEL ARCHITECTURES (formerly src/model.py)
# -----------------------------------------------------------------------------

def build_model(architecture: str, vectorizer=None):
    """Build one CNN + recurrent architecture with type and spam output heads."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {ARCHITECTURES}")

    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="message_content")
    priority_input = tf.keras.Input(shape=(), dtype=tf.string, name="priority")

    if vectorizer is None:
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=SEQ_LEN,
            name="text_vectorization",
        )

    tokens = vectorizer(text_input)
    # Keep mask_zero=False as in the consolidated submission implementation:
    # Conv1D does not propagate the Embedding mask. Padding-aware convolution
    # remains a documented limitation rather than silently changing the model.
    x = tf.keras.layers.Embedding(
        2000, EMBED_DIM, mask_zero=False, name="token_embedding"
    )(tokens)
    x = tf.keras.layers.Conv1D(
        64, 3, padding="same", activation="relu", name="text_cnn"
    )(x)
    x = tf.keras.layers.MaxPooling1D(2, name="text_pool")(x)

    if architecture == "CNN-BiGRU":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(RNN_UNITS), name="bigru"
        )(x)
    elif architecture == "CNN-BiLSTM":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(RNN_UNITS), name="bilstm"
        )(x)
    elif architecture == "CNN-StackLSTM":
        x = tf.keras.layers.LSTM(
            RNN_UNITS, return_sequences=True, name="stack_lstm_1"
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        x = tf.keras.layers.LSTM(RNN_UNITS // 2, name="stack_lstm_2")(x)
    else:  # CNN-BiRNN
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(RNN_UNITS), name="birnn"
        )(x)

    priority_lookup = tf.keras.layers.StringLookup(
        vocabulary=PRIORITIES, mask_token=None, num_oov_indices=1,
        name="priority_lookup"
    )
    p = priority_lookup(priority_input)
    p = tf.keras.layers.Embedding(
        len(PRIORITIES) + 1, 8, name="priority_embedding"
    )(p)
    p = tf.keras.layers.Flatten(name="priority_flatten")(p)

    fused = tf.keras.layers.Concatenate(name="context_fusion")([x, p])
    fused = tf.keras.layers.Dense(64, activation="relu", name="dense_features")(fused)
    fused = tf.keras.layers.Dropout(DROPOUT, name="feature_dropout")(fused)

    type_output = tf.keras.layers.Dense(
        len(CLASSES), activation="softmax", name="message_type"
    )(fused)
    spam_output = tf.keras.layers.Dense(1, activation="sigmoid", name="spam")(fused)

    model = tf.keras.Model(
        inputs={"message_content": text_input, "priority": priority_input},
        outputs={"message_type": type_output, "spam": spam_output},
        name=architecture.replace("-", "_"),
    )
    model.compile(
        optimizer="adam",
        loss={
            "message_type": "sparse_categorical_crossentropy",
            "spam": "binary_crossentropy",
        },
        metrics={"message_type": ["accuracy"], "spam": ["accuracy"]},
    )
    return model, vectorizer

# -----------------------------------------------------------------------------
# 4. PREDICTION (formerly src/predict.py)
# -----------------------------------------------------------------------------

def _unpack_predictions(output):
    """Support both dict outputs (newer models) and list/tuple outputs (older models)."""
    if isinstance(output, dict):
        type_probs = np.asarray(output["message_type"])
        spam_probs = np.asarray(output["spam"])
    elif isinstance(output, (list, tuple)) and len(output) == 2:
        type_probs = np.asarray(output[0])
        spam_probs = np.asarray(output[1])
    else:
        raise ValueError("Expected model outputs for message_type and spam.")
    return type_probs, spam_probs.reshape(-1)


def predict(model_path: str | Path, message: str, priority: str):
    """Load a saved model and return (class, confidence, spam label, spam probability)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    model = tf.keras.models.load_model(model_path, compile=False)
    output = model.predict(
        {
            "message_content": tf.constant([str(message)], dtype=tf.string),
            "priority": tf.constant([str(priority)], dtype=tf.string),
        },
        verbose=0,
    )
    type_probs, spam_probs = _unpack_predictions(output)
    class_index = int(np.argmax(type_probs[0]))
    spam_probability = float(spam_probs[0])
    return (
        CLASSES[class_index],
        float(type_probs[0][class_index]),
        "Yes" if spam_probability >= 0.5 else "No",
        spam_probability,
    )

# -----------------------------------------------------------------------------
# 5. TRAINING AND ARCHITECTURE COMPARISON (formerly src/train_compare.py)
# -----------------------------------------------------------------------------

def train_all(data_path: str, epochs: int = 5, output_dir: str = "models", batch_size: int = 32):
    tf.keras.utils.set_random_seed(SEED)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    train_df, val_df, test_df = prepare(df)
    print(
        f"Rows={len(df)} | Train={len(train_df)} | Validation={len(val_df)} | Test={len(test_df)}"
    )
    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df)
    x_test, y_test = xy(test_df)
    x_train, x_val, x_test = tensors(x_train), tensors(x_val), tensors(x_test)

    results = []
    for architecture in ARCHITECTURES:
        print(f"\n=== Training {architecture} ===")
        tf.keras.utils.set_random_seed(SEED)

        # Adapt the vocabulary only on training messages, then use this same
        # vectorizer instance in the model that is trained and saved.
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=SEQ_LEN,
            name="text_vectorization",
        )
        vectorizer.adapt(train_df["Message Content"].astype(str).values)
        model, _ = build_model(architecture, vectorizer=vectorizer)

        start = time.perf_counter()
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        runtime = time.perf_counter() - start

        raw_predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
        type_probs, spam_probs = _unpack_predictions(raw_predictions)
        predicted_type = np.argmax(type_probs, axis=1)
        predicted_spam = (spam_probs >= 0.5).astype("int32")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test["message_type"],
            predicted_type,
            average="macro",
            zero_division=0,
        )
        metrics = {
            "architecture": architecture,
            "type_accuracy": float(accuracy_score(y_test["message_type"], predicted_type)),
            "type_precision": float(precision),
            "type_recall": float(recall),
            "type_f1": float(f1),
            "spam_accuracy": float(accuracy_score(y_test["spam"], predicted_spam)),
            "runtime_seconds": float(runtime),
            "test_rows": int(len(test_df)),
        }
        results.append(metrics)

        model_file = output_path / f"{architecture.replace('-', '_')}.keras"
        model.save(model_file)
        evaluation = {
            "metrics": metrics,
            "message_type_classification_report": classification_report(
                y_test["message_type"],
                predicted_type,
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                output_dict=True,
                zero_division=0,
            ),
            "message_type_confusion_matrix": confusion_matrix(
                y_test["message_type"],
                predicted_type,
                labels=list(range(len(CLASSES))),
            ).tolist(),
        }
        (output_path / f"{architecture.replace('-', '_')}_evaluation.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))

    (output_path / "comparison_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nAll models and results saved in: {output_path.resolve()}")
    return results

# -----------------------------------------------------------------------------
# 6. MODEL DIAGNOSTIC / HELD-OUT EVALUATION (formerly Check_model.py)
# -----------------------------------------------------------------------------

def evaluate_saved_model(data_path: str, model_path: str):
    df = load_data(data_path)
    _, _, test_df = prepare(df)
    x_test, y_test = xy(test_df)
    x_test = tensors(x_test)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    model = tf.keras.models.load_model(model_path, compile=False)
    raw = model.predict(x_test, verbose=0)
    type_probs, spam_probs = _unpack_predictions(raw)

    y_true = np.asarray(y_test["message_type"], dtype=int)
    spam_true = np.asarray(y_test["spam"], dtype=int).ravel()
    if type_probs.ndim != 2 or type_probs.shape[1] != len(CLASSES):
        raise ValueError(f"Expected type probabilities shaped (N, {len(CLASSES)}), got {type_probs.shape}")
    if len(type_probs) != len(y_true) or len(spam_probs) != len(spam_true):
        raise ValueError("Prediction and target row counts do not match.")
    if not np.isfinite(type_probs).all() or not np.isfinite(spam_probs).all():
        raise ValueError("Model generated non-finite prediction values.")

    predicted_type = np.argmax(type_probs, axis=1)
    predicted_spam = (spam_probs >= 0.5).astype(int)
    print("Test rows:", len(test_df))
    print("True message type distribution:", Counter(y_true))
    print("True spam distribution:", Counter(spam_true))
    print("Predicted message type distribution:", Counter(predicted_type))
    print("Message type accuracy:", accuracy_score(y_true, predicted_type))
    print("Spam accuracy:", accuracy_score(spam_true, predicted_spam))
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, predicted_type, labels=list(range(len(CLASSES)))))
    print("\nClassification report:")
    print(classification_report(
        y_true, predicted_type, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0
    ))
    print("\nFirst 20 actual vs predicted:")
    for index in range(min(20, len(y_true))):
        actual, predicted = int(y_true[index]), int(predicted_type[index])
        print(
            f"{index + 1:02d}. Actual={CLASSES[actual]} | "
            f"Predicted={CLASSES[predicted]} | Correct={actual == predicted}"
        )
    print("\nType probabilities finite:", bool(np.isfinite(type_probs).all()))
    print("Type probability range:", float(type_probs.min()), float(type_probs.max()))
    print("Spam probability range:", float(spam_probs.min()), float(spam_probs.max()))

# -----------------------------------------------------------------------------
# 7. STREAMLIT APPLICATION (formerly app/app.py)
# -----------------------------------------------------------------------------

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="V2X Message Classification",
        page_icon="🚗",
        layout="wide",
    )
    st.title("🚗 V2X Message Classification — Architecture Comparison")
    st.caption("CNN-BiGRU baseline | CNN-BiLSTM | CNN-StackLSTM | CNN-BiRNN")
    st.info(
        "Presentation/demo interface. Research metrics must come from the verified "
        "dataset and held-out test set."
    )

    root = Path(__file__).resolve().parent
    model_dir = Path(os.environ.get("V2X_MODEL_DIR", str(root / "models_realistic")))
    message = st.text_area(
        "V2X Message",
        value="Vehicle collision ahead. Immediate emergency assistance required.",
    )
    priority = st.selectbox("Priority", PRIORITIES)
    architecture = st.selectbox("Architecture", ARCHITECTURES)

    if st.button("Analyze Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Enter a V2X message first.")
        else:
            model_path = model_dir / f"{architecture.replace('-', '_')}.keras"
            if not model_path.exists():
                st.error(f"Model not found: {model_path}. Train the selected architecture first.")
            else:
                try:
                    # Load once and use the same prediction for the metrics and
                    # the expandable class-probability breakdown.
                    loaded_model = tf.keras.models.load_model(model_path, compile=False)
                    raw = loaded_model.predict(
                        {
                            "message_content": tf.constant([message], dtype=tf.string),
                            "priority": tf.constant([priority], dtype=tf.string),
                        },
                        verbose=0,
                    )
                    type_probs, spam_probs = _unpack_predictions(raw)
                    class_index = int(np.argmax(type_probs[0]))
                    label = CLASSES[class_index]
                    confidence = float(type_probs[0][class_index])
                    spam_probability = float(spam_probs[0])
                    spam_label = "Yes" if spam_probability >= 0.5 else "No"
                    left, right = st.columns(2)
                    left.subheader("Message Type")
                    left.header(label)
                    left.metric("Confidence", f"{confidence * 100:.1f}%")
                    right.subheader("Spam")
                    right.header(spam_label)
                    right.metric("Spam probability", f"{spam_probability * 100:.1f}%")
                    with st.expander("View all class probabilities"):
                        ordered = sorted(
                            zip(CLASSES, type_probs[0]),
                            key=lambda item: float(item[1]),
                            reverse=True,
                        )
                        for class_name, probability in ordered:
                            st.write(f"{class_name}: {float(probability) * 100:.2f}%")
                            st.progress(float(probability))
                except Exception as error:
                    st.exception(error)

    st.subheader("Proposed Methodology")
    st.code(
        "V2X Message -> Tokenization -> Embedding -> CNN -> Recurrent Architecture\n"
        "                                            +\n"
        "                                     Priority Embedding\n"
        "                                            |\n"
        "                                      Context Fusion\n"
        "                                            |\n"
        "                                      Dense + Dropout\n"
        "                                      /             \\\n"
        "                           Message Type             Spam\n"
        "                              Softmax              Sigmoid"
    )
    st.caption(f"Model directory: {model_dir}")

# -----------------------------------------------------------------------------
# 8. COMMAND-LINE ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-file V2X CNN-recurrent architecture comparison project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and compare all four architectures.")
    train_parser.add_argument("--data", required=True, help="Path to the V2X CSV dataset.")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--out", default="models")
    train_parser.add_argument("--batch-size", type=int, default=32)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate one saved model on the held-out test split.")
    eval_parser.add_argument("--data", required=True, help="Path to the V2X CSV dataset.")
    eval_parser.add_argument("--model", required=True, help="Path to a saved .keras model.")

    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("--epochs and --batch-size must be positive integers.")
        train_all(args.data, args.epochs, args.out, args.batch_size)
    elif args.command == "evaluate":
        evaluate_saved_model(args.data, args.model)


if __name__ == "__main__":
    # Streamlit executes this file as __main__ without our custom subcommand.
    # Ordinary Python execution requires an explicit `train` or `evaluate` mode.
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "evaluate"}:
        main()
    else:
        # Streamlit's script runner does not consistently expose the word
        # "streamlit" in sys.argv, so detect its active execution context.
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
        except Exception:
            in_streamlit = False
        if in_streamlit:
            run_streamlit_app()
        else:
            print(
                "Choose how to run this file:\n"
                "  Streamlit UI: streamlit run V2X_All_In_One.py\n"
                "  Training:     python V2X_All_In_One.py train --data data/V2X_realistic_inspired_dataset_12000.csv --epochs 10 --out models_realistic\n"
                "  Evaluation:   python V2X_All_In_One.py evaluate --data data/V2X_realistic_inspired_dataset_12000.csv --model models_realistic/CNN_BiRNN.keras"
            )
