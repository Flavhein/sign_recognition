This project was done in the context of the seminar project of the course Multirate signal processing, during the summer semester 2025 in TU Ilmenau. gitlab.com/mrsp_project
I later forked and continued this project

## Sign Language recognition 

This project aim to build a model for sign prediction into text.
From the capture of the dataset to the training and tests.

### Files description

This folder contains: 
- the notebook ran on google colab to design and train the models
- the dataset build with the file capture.py
- the file sign\_language\_model.keras for the predictor model
- requirements.txt, only for the runner and capture.py
- the runner to test the model on your device

### How to run

In order to run "runner_predictor.py" or "capture.py"

```
python3 -m venv predictor_venv
source predictor_venv/bin/activate
pip install opencv-python numpy mediapipe tensorflow
python3 runner_predictor.py
```

The model constructed is a bidirectional LSTM, implemented using Tensorflow.
