# E2ERaga
This repository contains code for an end to end model for raga and tonic identification on audio samples

# Getting Started
Install the requirements by running `pip install -r requirements.txt`

## Model
1. Models can be downloaded from 
2. Place both the models in E2ERaga\model folder

## Run Time Input
E2ERaga supports audio samples which can be provided at runtime

Steps to run:
1. Run the command `python test_sample.py --runtime=True --tradition=hindustani --duration=60` 
2. You can change the tradition to hindustani/carnatic and duration to record in seconds
3. Once you run this command, there will be a prompt - `Press 1 to start recording or press 0 to exit:`
4. Enter accordingly and start recording for `duration` duration
5. After this the raga label and the tonic frequency is outputted




