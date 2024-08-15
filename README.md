# Real-Time Spoof Call Detection using Emotion Analysis

Don't hesitate to ⭐ the repo if you enjoy our work!

## In a nutshell

We developped a platform which can be used by emergency helplines to analyze the degree of genuineness and fakeness of a call recieved by them.

We analye vocal emotions, using deep learning based approaches. We deployed a web app using Flask

The tool can be accessed by installing the requirements and launching `main.py`.

## Table of Content :
- [I. Context](https://github.com/tanishqv010/Spoof-Call-Detection?tab=readme-ov-file#i-context)
- [II. Data Sources](https://github.com/tanishqv010/Spoof-Call-Detection?tab=readme-ov-file#ii-data-sources)
- [III. Methodology](https://github.com/tanishqv010/Spoof-Call-Detection?tab=readme-ov-file#iii-methodology)
- [IV. How to use it ?](https://github.com/tanishqv010/Spoof-Call-Detection?tab=readme-ov-file#iii-methodology)
- [V. Contributors](https://github.com/tanishqv010/Spoof-Call-Detection?tab=readme-ov-file#iii-methodology)

In this project, we are exploring state of the art models in multimodal sentiment analysis. We have chosen to explore text, sound and video inputs and develop an ensemble model that gathers the information from all these sources and displays it in a clear and interpretable way.

## 0. Technologies

![image](/static/techno.png)

## I. Context

Affective computing is a field of Machine Learning and Computer Science that studies the recognition and the processing of human voice.
Emotion Recognition is an old discipline but the one that aims to include sound is relatively new. This field has been rising with the development of social network that gave researchers access to a vast amount of data.

## II. Data Sources
We are using the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. This database contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity(normal, strong), with an additional neutral expression. All conditions are avail-able in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video(720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).” https://zenodo.org/record/1188976#.XCx-tc9KhQI

## III. Methodology
Our aim is to develop a model able to provide a live audio sentiment analysis with a visual user interface.

#### Pipeline

The speech emotion recognition pipeline was built the following way :
- Voice recording
- Audio signal discretization
- Log-mel-spectrogram extraction
- Split spectrogram using a rolling window
- Make a prediction using our pre-trained model

#### Model

The model we have chosen is a **Time Distributed Convolutional Neural Network**.

The main idea of a **Time Distributed Convolutional Neural Network** is to apply a rolling window (fixed size and time-step) all along the log-mel-spectrogram.
Each of these windows will be the entry of a convolutional neural network, composed by four Local Feature Learning Blocks (LFLBs) and the output of each of these convolutional networks will be fed into a recurrent neural network composed by 2 cells LSTM (Long Short Term Memory) to learn the long-term contextual dependencies. Finally, a fully connected layer with *softmax* activation is used to predict the emotion detected in the voice.

![image](/static/Pipeline.png)

To limit overfitting, we tuned the model with :
- Audio data augmentation
- Early stopping
- And kept the best model

<p align="center">
    <img src="/static/Accuracy%20Curve.png" width="400" height="400" />
</p>

## IV. How to use it ?

To use the web app :
- Clone the project locally
- Run `$ pip install -r requirements.txt`
- Launch `python main.py`
  
## V. Contributors

- <a href="https://github.com/anshikabharwal" title="Profile"> Anshika Bharwal
- <a href="https://github.com/krishnanarzary" title="Profile"> Krishna Narzary
- <a href="https://github.com/tanishqv010" title="Profile"> Tanishq Verma
