from fastapi import FastAPI, UploadFile, File
# We import FER here, but we WON'T initialize it yet
from fer import FER
import cv2
import numpy as np
import uvicorn

app = FastAPI()

# Global variable to hold the AI brain later
detector = None

@app.get("/")
def home():
    return {"message": "Mood Scanner API is running!"}

@app.post("/analyze")
async def analyze_mood(file: UploadFile = File(...)):
    global detector
    
    # 🧠 LAZY LOADING: Only load the brain if it's not ready
    if detector is None:
        print("Loading AI Model for the first time... (This might take a few seconds)")
        detector = FER(mtcnn=True)

    try:
        # 1. Read the image file sent by Flutter
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Analyze the image
        result = detector.detect_emotions(img)

        # 3. Logic: If no face detected
        if not result:
            return {"mood": "No Face Detected", "score": 0.0}

        # 4. Get the strongest emotion
        emotions = result[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        score = emotions[dominant_emotion]

        # 5. Return JSON to Flutter
        return {
            "mood": dominant_emotion.capitalize(),
            "score": float(score),
            "all_data": emotions
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)









# from fastapi import FastAPI, UploadFile, File
# from fer import FER
# import cv2
# import numpy as np
# import uvicorn

# app = FastAPI()

# # Initialize the AI Model (FER - Facial Expression Recognition)
# detector = FER(mtcnn=True) 

# @app.get("/")
# def home():
#     return {"message": "Mood Scanner API is running!"}

# @app.post("/analyze")
# async def analyze_mood(file: UploadFile = File(...)):
#     try:
#         # 1. Read the image file sent by Flutter
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # 2. Analyze the image
#         result = detector.detect_emotions(img)

#         # 3. Logic: If no face detected
#         if not result:
#             return {"mood": "No Face Detected", "score": 0.0}

#         # 4. Get the strongest emotion
#         emotions = result[0]["emotions"]
#         dominant_emotion = max(emotions, key=emotions.get)
#         score = emotions[dominant_emotion]

#         # 5. Return JSON to Flutter
#         return {
#             "mood": dominant_emotion.capitalize(),
#             "score": float(score),
#             "all_data": emotions
#         }

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)