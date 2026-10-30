V2X CONSOLIDATED PROJECT

Folder contents:
- V2X_All_In_One.py: consolidated Python source (training, evaluation, prediction, Streamlit UI)
- models_realistic/: your existing trained Keras models and comparison_results.json (copied unchanged)
- data/: project CSV datasets
- requirements.txt: Python dependencies

IMPORTANT:
Keep V2X_All_In_One.py, models_realistic/, and data/ together in this folder.
The Streamlit interface loads the existing models from models_realistic/ by default.
The model files are not embedded in the Python file and are not retrained when you launch the interface.

Install dependencies:
    pip install -r requirements.txt

Launch the existing-model demo:
    streamlit run V2X_All_In_One.py

Train all architectures (this writes updated model files to models_realistic/):
    python V2X_All_In_One.py train --data data/V2X_realistic_inspired_dataset_12000.csv --epochs 10 --out models_realistic

Evaluate one saved model:
    python V2X_All_In_One.py evaluate --data data/V2X_realistic_inspired_dataset_12000.csv --model models_realistic/CNN_BiRNN.keras

To use a different existing model folder without moving it, set V2X_MODEL_DIR to that folder before launching Streamlit.
