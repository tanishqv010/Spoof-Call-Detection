# Real-Time Spoof Call Detection using Emotion Analysis

## In a nutshell

We developped a platform which can be used by emergency helplines to analyze the degree of genuineness and fakeness of a call recieved by them.

We analye vocal emotions, using deep learning based approaches. We deployed a web app using Flask

The tool can be accessed by installing the requirements and launching `main.py`.

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

![image](<img width="1213" alt="sound_pipeline" src="https://github.com/tanishqv010/Spoof-Call-Detection/assets/70789222/c85c485f-320c-4cfb-85a6-3017bd994275">)

To limit overfitting, we tuned the model with :
- Audio data augmentation
- Early stopping
- And kept the best model

<p align="center">
    <img src="<img width="300" alt="Accuracy_Speech" src="https://github.com/tanishqv010/Spoof-Call-Detection/assets/70789222/808b8d1b-ea56-4425-8b29-cd0a21c290fc">
" width="400" height="400" />
</p>

## IV. How to use it ?

To use the web app :
- Clone the project locally
- Run `$ pip install -r requirements.txt``
- Launch `python main.py`
  
## V. Contributors

<table><tr><td align="center">
	<a href="https://github.com/anshikabharwal"> <!--     Github link -->
	<img src="" width="100px;" alt="Anshika Bharwal"/> <!--     Image link -->
	<br />
	<sub><b>Anshika Bharwal</b></sub>
	</a></td>
	<td align="center">
	<a href="https://github.com/krishnanarzary"> <!--     Github link -->
	<img src="" width="100px;" alt="Krishna Narzary"/> <!--     Image link -->
	<br />
	<sub><b>Krishna Narzary</b></sub>
	</a></td>
  <td align="center">
	<a href="https://github.com/tanishqv010"> <!--     Github link -->
	<img src="" width="100px;" alt="Tanishq Verma"/> <!--     Image link -->
	<br />
	<sub><b>Tanishq Verma</b></sub>
	</a></td>
</tr></table>
