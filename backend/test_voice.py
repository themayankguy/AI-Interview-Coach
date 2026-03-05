import sounddevice as sd
import numpy as np
import queue
import threading
import librosa
from faster_whisper import WhisperModel
import time

model = WhisperModel("base", compute_type="int8")
audio_queue = queue.Queue()

SAMPLE_RATE = 44100 #16000 nromally, but 44100 is more common for live audio
session_start_time = time.time()
total_words = 0
total_fillers = 0
filler_freq = {}
all_rms_values = []

voice_metrics = {
    "wpm": 0,
    "volume_stability": 0,
    "filler_count": 0,
    "filler_freq": {}
}

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

total_speaking_time = 0  # ADD GLOBAL AT TOP

def analyze_audio_chunk(audio_chunk):
    global voice_metrics, total_words, total_fillers, filler_freq
    global all_rms_values, total_speaking_time

    y = audio_chunk.flatten().astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Ignore silence
    energy = np.mean(np.abs(y))
    if energy < 0.01:
        return

    # Track speaking duration
    chunk_duration = len(y) / SAMPLE_RATE
    total_speaking_time += chunk_duration

    # Volume stability
    rms = librosa.feature.rms(y=y)[0]
    all_rms_values.extend(rms)

    # Transcription
    # Resample to 16000 for Whisper
    y_resampled = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=16000)

    segments, _ = model.transcribe(
        y_resampled,
        beam_size=1,
        vad_filter=True
        )

    text = ""
    for segment in segments:
        text += segment.text

    #print("TEXT:", text)  # debug

    words = len(text.split())
    total_words += words

    fillers = ["um", "uh", "like", "you know", "so", "actually","basically", "literally", "right", "okay","i mean", "kind of", "sort of", "well"]
    current_text_lower = text.lower()
    for f in fillers:
        count = current_text_lower.count(f)
        if count > 0:
            total_fillers += count
            filler_freq[f] = filler_freq.get(f, 0) + count

    # WPM based on speaking time
    if total_speaking_time > 0:
        wpm = (total_words / total_speaking_time) * 60
    else:
        wpm = 0

    # Stable bounded formula
    mean_volume = np.mean(all_rms_values)
    volume_std = np.std(all_rms_values)

    stability = 1 / (1 + (volume_std / (mean_volume + 1e-6)))

    voice_metrics["wpm"] = wpm
    voice_metrics["volume_stability"] = stability
    voice_metrics["filler_count"] = total_fillers
    voice_metrics["filler_freq"] = dict(filler_freq)


def reset_voice_metrics():
    global voice_metrics, total_words, total_fillers, filler_freq, all_rms_values, total_speaking_time
    voice_metrics["wpm"] = 0
    voice_metrics["volume_stability"] = 0
    voice_metrics["filler_count"] = 0
    voice_metrics["filler_freq"] = {}
    total_words = 0
    total_fillers = 0
    filler_freq = {}
    all_rms_values = []
    total_speaking_time = 0

def voice_thread():
    buffer = []
    samples_collected = 0
    samples_needed = SAMPLE_RATE * 4  # 4 seconds

    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE):

        while True:
            data = audio_queue.get()
            buffer.append(data)
            samples_collected += len(data)

            if samples_collected >= samples_needed:
                audio_chunk = np.concatenate(buffer)
                analyze_audio_chunk(audio_chunk)

                buffer = []
                samples_collected = 0

def start_voice_analysis():
    thread = threading.Thread(target=voice_thread)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    print("Starting live voice tracking...")
    start_voice_analysis()

    while True:
        time.sleep(1)
        print("\n--- Live Voice Metrics ---")
        print(f"WPM: {voice_metrics['wpm']:.0f}")
        print(f"Stability: {voice_metrics['volume_stability']*100:.1f}%")
        print(f"Fillers: {voice_metrics['filler_count']}")