import cv2
from deepface import DeepFace
import webbrowser

# 1. Define Music Folders or Links based on Emotions
music_map = {
    "happy": "https://www.youtube.com/results?search_query=happy+upbeat+songs",
    "sad": "https://www.youtube.com/results?search_query=lofi+chill+music",
    "angry": "https://www.youtube.com/results?search_query=calm+instrumental+music",
    "surprise": "https://www.youtube.com/results?search_query=energetic+party+songs",
    "neutral": "https://www.youtube.com/results?search_query=top+hits+2024"
}

# 2. Start Webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to capture emotion and get music!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live feed
    cv2.imshow('Emotion Detector - Press Q to Recommend', frame)

    # Press 'q' to analyze and recommend
    if cv2.waitKey(1) & 0xFF == ord('q'):
        try:
            # Analyze Emotion
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Get the dominant emotion
            emotion = results[0]['dominant_emotion']
            print(f"Detected Emotion: {emotion}")

            # Open Music Link
            if emotion in music_map:
                webbrowser.open(music_map[emotion])
            else:
                webbrowser.open(music_map["neutral"])
                
            break # Exit after recommendation
        except Exception as e:
            print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()