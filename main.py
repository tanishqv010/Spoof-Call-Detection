
import numpy as np
from speech_emotion_recognition import speechEmotionRecognition
from pydub import AudioSegment
import time
import keyboard
import multiprocessing
import json

# Voice Recording
def record(rec_duration, rec_sub_dir):
    # Instanciate new AudioRecorder object
    SER = speechEmotionRecognition()

    # Voice Recording
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    return


# Audio Emotion Analysis
def audio_dash(rec_sub_dir, emotion_list):
    # Sub dir to speech emotion recognition model
    model_sub_dir = r'audio.hdf5'

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Calculate emotion distribution
    emotion_dist = [(emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    emotion_dist = np.array(emotion_dist)

    emotion_dist = np.add(emotion_dist, emotion_list)


# trimming the Audio
def trim_audio_file(input_file, output_file, start_time, end_time):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Trim the audio file
    trimmed_audio = audio[start_time * 1000:end_time * 1000]

    # Export the trimmed audio file
    trimmed_audio.export(output_file, format="wav")


if __name__ == '__main__':
    rec_duration = 10 # Default Recording Duration
    on_call = True # Start Call
    recording_chunk = 0 # Default Recording Chunk
    analyse_chunk = 0 # Default Analyse Chunk
    emotion_list = np.array()
    process = []
    recording = multiprocessing.Process(target=record, args=[rec_duration, f"Recording\{recording_chunk}.wav"])
    analyse = multiprocessing.Process(target=audio_dash, args=[f"Recording\{analyse_chunk}.wav"])

    while(on_call):
        with open('on_call.json', 'r') as f:
            on_call = json.load(f)
            f.close()
        recording = multiprocessing.Process(target=record, args=[rec_duration, f"Recording\{recording_chunk}.wav"])
        analyse = multiprocessing.Process(target=audio_dash, args=[f"Recording\{analyse_chunk}.wav"])
        if not keyboard.is_pressed('e'):
            recording.start()
            recording_chunk += 1
            recording.join()
        else:
            on_call = False # End Call
        analyse.start()
        process.append(analyse)
        analyse_chunk += 1

    for i in range(analyse_chunk, recording_chunk + 1):
        analyse = multiprocessing.Process(target=audio_dash, args=[f"Recording\{i}.wav", emotion_list])
        analyse.start()

    for prc in process:
        prc.join()

    with open('on_call.json', 'w') as f:
        f.write('\"True\"')
        f.close()

    print("Call Ended")

    emotion_list = np.divide(emotion_list, analyse_chunk)
    with open("emotion_dist.json", "a") as file:
        file.write(f"{str(emotion_list)}\n")
        file.close()

    # Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    emotion_weights = np.array([-0.3, 0, -1, 0.8, 0.5, -0.5, 0])
    spoofness = np.dot(emotion_weights, emotion_list)
    print(spoofness)