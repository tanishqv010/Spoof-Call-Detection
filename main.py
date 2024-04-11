from flask import Flask, render_template, request, jsonify
from speech_emotion_recognition import speechEmotionRecognition
from pydub import AudioSegment
import time
import json
import numpy as np

app = Flask(__name__)

# Voice Recording
def record(rec_duration, rec_sub_dir):
    SER = speechEmotionRecognition()
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

# Audio Emotion Analysis
def audio_dash(rec_sub_dir):
    model_sub_dir = r'audio.hdf5'
    SER = speechEmotionRecognition(model_sub_dir)
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)
    emotion_dist = [(emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    return emotion_dist

# Add two emotion distributions
def add(emo1, emo2):
    return [emo1[i] + emo2[i] for i in range(0, len(emo1))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    rec_duration = 10 # Default Recording Duration
    on_call = True # Start Call
    recording_chunk = 0 # Default Recording Chunk
    emo_tot = [0] * 7 # Default Emotion Distribution

    while True:
        with open('call.json', 'r') as f:
            on_call = int(json.load(f))
            f.close()
        if on_call == 0:
            break
        else:
            record(rec_duration, f"Recording/{recording_chunk}.wav")
            recording_chunk += 1

    for analyse_chunk in range(0, recording_chunk):
        emo_curr = audio_dash(f"Recording/{analyse_chunk}.wav")
        emo_tot = add(emo_tot, emo_curr)

    emo_avg = [emo / recording_chunk for emo in emo_tot]
    with open('emotion_dist.json', 'w') as f:
        json.dump({emotion: (emo_avg[i]) for i, emotion in enumerate(SER._emotion.values())}, f)
        f.close()

    with open('call.json', 'w') as f:
        json.dump(1, f)
        f.close()

    emo_wt = np.array([-0.3, 0, -1, 0.8, 0.5, -0.5, 0])
    spoofness = np.dot(emo_wt, emo_avg)
    print(spoofness) # Print degree of spoofness on website

    return jsonify({'message': 'Process started successfully.'})

@app.route('/stop', methods=['POST'])
def stop():
    with open('call.json', 'w') as f:
        json.dump(0, f)
        f.close()
    return jsonify({'message': 'Process stopped successfully.'})

if __name__ == '__main__':
    app.run(debug=True)