This project was done in the context of the seminar project of the course Multirate signal processing, during the summer semester 2025 in TU Ilmenau.

## AI-Based Gesture Animation for Sign Language Avatars

This project aim to build two models, one for sign prediction into text, another one for sign generation from text. In order to produce realistic sign gesture animation for Avatars.

### Files description

This folder contains: 

-the notebook (two) ran on google colab to design and train the models

-the dataset build with the file capture.py

-the report and the slide show of the project

-the file cond_gen.pt for the generator model

-the file sign\_language\_model.keras for the predictor model

-requirements.txt, only for the runner and capture.py

-the two runner to test the models on your device

### How to run

In order to run "runner_generator.py" 
here are the steps :

in your cmd on linux :

```
python3 -m venv animator_venv
source animator_venv/bin/activate
pip install torch numpy matplotlib IPython
python3 runner_generator.py
```


In order to run "runner_predictor.py" or "capture.py"

```
python3 -m venv predictor_venv
source predictor_venv/bin/activate
pip install opencv-python numpy mediapipe tensorflow
python3 runner_predictor.py
```

Flavien ROMANETTI

73100