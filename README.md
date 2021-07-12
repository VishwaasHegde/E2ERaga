# E2ERaga
This repository contains code for an end to end model for raga and tonic identification on audio samples


A better user-friendly repositiory containing only the inference code can be accessed here - https://github.com/VishwaasHegde/CRETORA

# Getting Started
Requires `python==3.6.9`

Download and install [Anaconda](https://www.anaconda.com/products/individual) for easier package management

Install the requirements by running `pip install -r requirements.txt`

## Model
1. Create an empty folder called `model` and place it in E2ERaga folder
2. Download the models (model-full.h5 and hindustani_raga_model.hdf5) from [here](https://drive.google.com/drive/u/0/folders/1n5u8jHsFUET2krAj1JDAT1dwjMLKA6mT) and place them in models folder 

## Data
1. I dont have the permisssion to upload the datasets, the datasets has to be obtained by request from here: https://compmusic.upf.edu/node/328
2. Or contact me directly: vishwaas (dot) universe (at) gmail (dot) com

## Run Time Input
E2ERaga supports audio samples which can be recorded at runtime

Steps to run:
1. Run the command `python test_sample.py --runtime=True --tradition=hindustani --duration=60` 
2. You can change the tradition to hindustani/carnatic and duration to record in seconds
3. Once you run this command, there will be a prompt - `Press 1 to start recording or press 0 to exit:`
4. Enter accordingly and start recording for `duration` duration
5. After this the raga label and the tonic frequency is outputted

## File input
E2ERaga supports recorded audio samples which can be provided at runtime

Steps to run:
1. Run the command `python test_sample.py --runtime_file=<audio_file_path> --tradition=<hindustani/carnatic> --file_type=wav`
   
   Example: `python test_sample.py --runtime_file=data/sample_data/Ahira_bhairav_27.wav --tradition=hindustani --file_type=wav`
3. The model supports wav and mp3 file (--filetype defaults to wav if not provided), with mp3 there will be a delay in converting into wav format internally
4. After this the raga label and the tonic frequency is outputted

Demo videos:

## Live Raga Prediction for Raga: Miyan Malhar

(Click will redirect to youtube)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/GC1MyaENUOc/0.jpg)](https://www.youtube.com/watch?v=GC1MyaENUOc)


## Live Raga Prediction for Raga: Des


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/K4Kv_Pypk1w/0.jpg)](https://www.youtube.com/watch?v=K4Kv_Pypk1w)


### Hindustani Raga Embedding cosine similarity obtained from the model

![alt text](https://github.com/VishwaasHegde/CRETORA/blob/main/images/hindustani_weights.png)

### Carnatic Raga Embedding cosine similarity obtained from the model

![alt text](https://github.com/VishwaasHegde/CRETORA/blob/main/images/carnatic_weights.png)

Acknowledgments:
1. The model uses [CREPE](https://github.com/marl/crepe) to find the pitches for the audio, I would like to thank [Jong Wook](https://github.com/jongwook) for clarifiying my questions
2. Also thank [CompMusic](https://compmusic.upf.edu/node/328) and Sankalp Gulati for providing me the datasets


Acknowledgments:
1. The model uses [CREPE](https://github.com/marl/crepe) to find the pitches for the audio, I would like to thank [Jong Wook](https://github.com/jongwook) for clarifiying my questions
2. Also thank [CompMusic](https://compmusic.upf.edu/node/328) and Sankalp Gulati for providing me the datasets


