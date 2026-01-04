# import pyttsx3
# import os

# engine = pyttsx3.init()
# engine.setProperty('rate', 160)
# engine.setProperty('volume', 1.0)

# alerts = {
#     "audio/phase1_warning.wav":
#     "Please stay alert. You appear slightly drowsy.",

#     "audio/phase2_warning.wav":
#     "Warning. You are showing signs of drowsiness. Please concentrate.",

#     "audio/phase3_danger.wav":
#     "Critical alert! You are extremely drowsy. Stop the vehicle immediately."
# }

# os.makedirs("audio", exist_ok=True)

# for path, text in alerts.items():
#     engine.save_to_file(text, path)

# engine.runAndWait()
# print("Audio files generated successfully")


import pyttsx3
import os

engine = pyttsx3.init()

# Optional tuning
engine.setProperty('rate', 160)     # speaking speed
engine.setProperty('volume', 1.0)   # volume (0.0 – 1.0)

# Ensure audio folder exists
os.makedirs("audio", exist_ok=True)

alerts = {
    "audio/head_left.wav":  "Please keep your head straight. Do not tilt left.",
    "audio/head_right.wav": "Please keep your head straight. Do not tilt right.",
    "audio/head_front.wav": "Please do not bend forward while driving.",
    "audio/head_back.wav":  "Please avoid leaning backward while driving."
}

for path, text in alerts.items():
    engine.save_to_file(text, path)

engine.runAndWait()
print("Head tilt audio files generated successfully.")
