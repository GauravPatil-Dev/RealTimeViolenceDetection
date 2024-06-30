# Real-Time Violence Detection System

This project implements a real-time violence detection system using an LSTM model. It includes a web application that allows users to either upload a video or analyze violence in real-time using a camera. The results, including an analysis plot, are displayed on the UI.

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Business Use Case](#business-use-case)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Move Files](#step-1-move-files)
  - [Step 2: Extract Files](#step-2-extract-files)
  - [Step 3: Extract Features](#step-3-extract-features)
  - [Step 4: Train Model](#step-4-train-model)
  - [Step 5: Run Web Application](#step-5-run-web-application)
- [License](#License)
## Introduction

The system processes video footage to detect violent activities in real-time. It uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN - LSTM) to classify video sequences and compute a mean violence score for each sequence.

## Architecture

![Architecture](architecture.png)

## Business Use Case

![Business Use Case](use_case.png)

## Installation

To get started, clone the repository and install the required dependencies:

`git clone <repository-url>`

`cd <repository-directory>`

`pip install -r requirements.txt`

### Install `ffmpeg` for Frame Extraction
FFmpeg is required for extracting frames from videos. Follow the instructions below to install FFmpeg on your system:
  ### On Ubuntu/Debian:

  `sudo apt update`

  `sudo apt install ffmpeg`

  ### On macOS using Homebrew:

  `brew install ffmpeg`

  ### On Windows:

  Download ffmpeg from ffmpeg.org

  Extract the downloaded zip file.

  Add the bin folder from the extracted files to your system's PATH.

## Step 1: Move Files
Navigate to the lstm_model directory and run the 1_move_file.py script to move all files into the appropriate train/test folders.

## Step 2: Extract Files
Run the 2_extract_files.py script to extract images from the videos and create a data file for training and testing.

## Step 3: Extract Features
Navigate to the data directory inside the lstm_model folder and run extract_features.py to generate extracted features for each video.

## Step 4: Train Model
In the same data directory, run train.py to train the LSTM model. The trained model will be saved in the checkpoints directory.

## Step 5: Run Web Application
Navigate to the root directory of the project and run main.py to start the web application:
`python main.py`

Ensure the saved_model path variable in main.py points to the correct location of the trained model

## License
```markdown
MIT License

Copyright (c) 2024 GauravPatil-Dev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
