"""
Iris Flower Classifier - Prediction Script
==========================================
Run this after train.py to classify new flower measurements.
Usage:
    python predict.py                          # interactive mode
    python predict.py 5.1 3.5 1.4 0.2         # pass measurements directly
"""

import sys
import numpy as np
import joblib

FEATURE_NAMES  = ["sepal length (cm)", "sepal width (cm)",
                  "petal length (cm)", "petal width (cm)"]
SPECIES_NAMES  = ["setosa", "versicolor", "virginica"]
SPECIES_EMOJI  = {"setosa": "🌸", "versicolor": "🌺", "virginica": "🌼"}

MODEL_PATH  = "outputs/best_model.pkl"
SCALER_PATH = "outputs/scaler.pkl"


def load_artifacts():
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        print("❌  Model not found. Please run  train.py  first.")
        sys.exit(1)


def get_measurements_interactive() -> list[float]:
    print("\n🌸  Iris Flower Classifier — Prediction")
    print("─" * 40)
    print("Enter flower measurements (in cm):\n")
    values = []
    for feat in FEATURE_NAMES:
        while True:
            try:
                val = float(input(f"  {feat}: "))
                values.append(val)
                break
            except ValueError:
                print("    ⚠  Please enter a numeric value.")
    return values


def predict(measurements: list[float], model, scaler) -> dict:
    X = np.array(measurements).reshape(1, -1)
    X_sc = scaler.transform(X)
    species_idx = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]
    return {
        "species":     SPECIES_NAMES[species_idx],
        "confidence":  proba[species_idx],
        "all_proba":   dict(zip(SPECIES_NAMES, proba)),
    }


def display_result(measurements: list[float], result: dict):
    sp   = result["species"]
    conf = result["confidence"]
    emoji = SPECIES_EMOJI.get(sp, "🌿")

    print("\n" + "─" * 40)
    print(f"  Prediction:  {emoji}  Iris {sp.capitalize()}")
    print(f"  Confidence:  {conf:.1%}")
    print("\n  Class probabilities:")
    for cls, prob in result["all_proba"].items():
        bar = "█" * int(prob * 20)
        print(f"    {cls:<12}  {bar:<20}  {prob:.1%}")
    print("─" * 40)

    # Typical ranges for reference
    print("\n  📏  Measurement summary:")
    for feat, val in zip(FEATURE_NAMES, measurements):
        print(f"    {feat:<22} {val:.1f} cm")
    print()


if __name__ == "__main__":
    model, scaler = load_artifacts()

    if len(sys.argv) == 5:
        # CLI mode: python predict.py 5.1 3.5 1.4 0.2
        try:
            measurements = [float(v) for v in sys.argv[1:]]
        except ValueError:
            print("❌  All four arguments must be numeric.")
            sys.exit(1)
    else:
        measurements = get_measurements_interactive()

    result = predict(measurements, model, scaler)
    display_result(measurements, result)
