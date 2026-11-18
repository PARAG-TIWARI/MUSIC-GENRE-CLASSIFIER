# Music Genre Classification with CNN

This project is a music-genre classification prototype built around a convolutional neural network (CNN). The current Streamlit interface accepts a **spectrogram image** and predicts one of ten music genres using a saved Keras model.

It is an image-based inference demo. The current web interface does not accept an audio file and does not convert audio into a spectrogram at prediction time.

## Current application

**Main entry point:** \`app.py\`  
**Model used by the app:** \`model/cnn_genre_model.h5\`

The app:
1. Loads the saved Keras model once using Streamlit's resource cache.
2. Accepts PNG, JPG or JPEG image input.
3. Converts the image to RGB.
4. Resizes it to 128 × 128 pixels.
5. Scales pixel values to the 0–1 range.
6. Runs CNN inference and displays the predicted genre and the highest softmax score.

The class order in the app is:

\`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock\`

This is the alphabetical order used by Keras' directory-based image generator for the matching class folders. Keep the folder names and the class order aligned if the model is retrained.

## Technology

- Python 3.10
- TensorFlow / Keras
- CNN image classification
- NumPy and Pillow
- Streamlit

## Run locally

Create a Python 3.10 environment:

~~~bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
~~~

Install the pinned dependencies:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Start the app:

~~~bash
streamlit run app.py
~~~

Open the local Streamlit address shown in the terminal (normally \`http://localhost:8501\`).

The checked-in \`runtime.txt\` specifies Python 3.10. The repository does not currently record a verified public Streamlit deployment URL, so this project should be treated as local-run unless a current hosting URL is added and verified.

## Model training

The active CNN training script is \`src/train_cnn.py\`.

It currently:
- Reads class folders from \`Data/images_original\`.
- Uses \`ImageDataGenerator\` with pixel rescaling and a 20% validation split.
- Resizes images to 128 × 128.
- Uses batch size 32 and a maximum of 25 epochs.
- Uses early stopping with patience 5 and restores the best validation weights.
- Saves the model to \`model/cnn_genre_model.h5\`.

Run from the repository root:

~~~bash
python src/train_cnn.py
~~~

Training requires the spectrogram image folders under \`Data/images_original\`. The current script uses a training/validation split only; it does not create an independent held-out test set or write a formal evaluation report. Do not quote accuracy or performance metrics unless you have retained a reproducible test run, split definition and output.

## Repository structure

~~~text
MUSIC-GENRE-CLASSIFIER/
├── app.py                         # Streamlit CNN inference app
├── model/
│   └── cnn_genre_model.h5         # Saved CNN model (~75 MB)
├── src/
│   ├── train_cnn.py               # Active CNN training script
│   ├── predict_cnn.py             # Legacy SVM prediction experiment (see note below)
│   ├── train_svm_30sec.py.py      # Legacy SVM training experiment
│   └── train_svm_3sec.py           # Incomplete SVM experiment
├── Data/
│   ├── images_original/           # Spectrogram images used by CNN training
│   ├── genres_original/           # WAV audio files
│   ├── features_30_sec.csv        # Precomputed audio features
│   └── features_3_sec.csv         # Precomputed audio features
├── requirements.txt
└── runtime.txt
~~~

## Legacy SVM experiment status

The SVM-related files are not the inference path used by \`app.py\` and should be considered **experimental / unfinished**:

- \`src/train_svm_30sec.py.py\` reads from a \`dataset/\` directory, while the committed feature tables are under \`Data/\`.
- \`src/train_svm_3sec.py\` contains only the initial data-loading and feature/label selection statements.
- \`src/predict_cnn.py\` expects \`model/genre_model.pkl\` and \`model/scaler.pkl\`, but those files are not present in the repository. The 30-second SVM script writes different filenames.

Use the saved CNN model and \`app.py\` as the supported project path unless the SVM experiment is repaired and evaluated separately.

## Dataset and repository size

The current Git tree contains approximately:
- 1,000 WAV audio files under \`Data/genres_original\`.
- 999 spectrogram images under \`Data/images_original\`.
- Two feature CSV files.
- A saved CNN model of approximately 75 MB.

The repository therefore has a large clone footprint (around 1.4 GB of Git blobs in the current tree). For a more maintainable public portfolio, consider moving raw audio and generated spectrogram datasets to a documented external storage location or dataset download step, and using Git LFS or versioned model releases for large model artifacts. The existing tracked files have not been deleted or rewritten by this documentation update.

## Evaluation and limitations

The current repository does not include:
- A held-out test evaluation script for the CNN model.
- A committed metrics report or confusion matrix.
- A dataset provenance/license note.
- A class-index artifact generated by the training script.
- An automated test suite for preprocessing or inference.

The displayed softmax maximum is a model score, not a calibrated probability that the prediction is correct. Results may vary with spectrogram generation method, colour scale, axis layout, crop and image resolution. The current app expects a compatible spectrogram representation.

Before claiming model accuracy, document the dataset source and licence, exact split, training configuration, random seed, model checkpoint, test metrics and confusion matrix.

## Dataset attribution and licence

The repository currently does not contain a clear dataset provenance/licence statement or a project-level LICENSE file. Confirm the original source and permitted redistribution terms for the WAV files and generated images before redistributing the data or claiming commercial-use rights.

## Future improvements

1. Add a reproducible audio-to-spectrogram preprocessing pipeline so users can submit audio rather than pre-generated images.
2. Persist class names alongside the model artifact to avoid label-order drift.
3. Add a held-out test set and report macro-F1, per-class precision/recall, confusion matrix and accuracy.
4. Record dataset provenance, licence and split information.
5. Separate raw data and generated artifacts from the source-code repository.
6. Remove or repair the incomplete SVM scripts so the repository exposes one clear, supported workflow.

## Author

Parag Tiwari · [GitHub profile](https://github.com/PARAG-TIWARI)
