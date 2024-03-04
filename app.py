from flask import Flask, render_template, request
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
import cv2
import pandas as pd
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import pickle
import matplotlib.pyplot as plt

# model = pickle.load(open('model.pkl', "rb"))
app = Flask(__name__)

# Set up YouTube

model_path = 'sentiment_analysis_model.pkl'

@app.route("/")
def index():
    return render_template('dashboard.html')

@app.route("/url", methods=['POST'])
def step():
    video_url = request.form.get('url')
    output_path = "C:\\Users\\shail\\Dropbox\\My PC (LAPTOP-674MEPPR)\\Desktop\\Final Year\\Code"
    download_video(video_url, output_path)
    download_transcription(video_url, output_path)
    download_comments_from_youtube(video_url, API_KEY, output_path)
    return "Completed!!"

def get_video_id(video_url):
    query = urlparse(video_url)
    if query.hostname == "www.youtube.com":
        if "v" in query.query:
            video_id = parse_qs(query.query)["v"][0]
            return video_id
    elif query.hostname == "youtu.be":
        video_id = query.path[1:]
        return video_id
    return None

# import os
# from pytube import YouTube

def download_video(video_url, output_path='.'):
    try:
        # Create a YouTube object
        youtube = YouTube(video_url)

        # Get the highest resolution stream
        video_stream = youtube.streams.get_highest_resolution()

        # Set the desired video name
        video_name = "video.mp4"

        # Create a folder with the video ID as the name
        video_id = youtube.video_id
        folder_path = os.path.join(output_path, video_id)
        os.makedirs(folder_path, exist_ok=True)

        # Set the complete file path, including the video name
        file_path = os.path.join(folder_path, video_name)

        # Download the video to the specified output path with the desired name
        video_stream.download(output_path=folder_path, filename=video_name)

        print(f"Download complete! Video saved to: {file_path}")

        convert_video_to_frames(file_path)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# download_video("https://www.youtube.com/watch?v=example_video_id", "output_directory")


def download_transcription(video_url, output_path='.'):
    try:
        # Create a YouTube object
        youtube = YouTube(video_url)

        # Get the video ID
        video_id = youtube.video_id

        # Get the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Create a folder with the video ID as the name
        folder_path = os.path.join(output_path, video_id)
        os.makedirs(folder_path, exist_ok=True)

        # Save the transcript to a file in the folder
        output_trans = os.path.join(folder_path, f"transcription.txt")
        with open(output_trans, 'w', encoding='utf-8') as file:
            for entry in transcript:
                file.write(f"{entry['text']}\n\n")

        print(f"Transcription saved to: {output_trans}")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_comments_from_youtube(video_url, api_key, output_path='.'):
    try:
        # Get YouTube video ID from the URL
        video_id = get_video_id(video_url)

        if not video_id:
            print("Invalid YouTube video URL.")
            return

        # Authenticate and create the YouTube API service
        youtube = get_authenticated_service(api_key)

        # Get comments for the specified video
        comments = get_video_comments(
            youtube,
            video_id,
            part="snippet",
            textFormat="plainText"
        )

        # Create a folder with the video ID as the name
        folder_path = os.path.join(output_path, video_id)
        os.makedirs(folder_path, exist_ok=True)

        # Save comments to a CSV file in the folder
        output_file = os.path.join(folder_path, "comments.csv")
        save_comments_to_csv(comments, output_file)
        rest = comments_analysis(output_file, model_path)

        return rest

    except Exception as e:
        print(f"An error occurred: {e}")

def get_authenticated_service(api_key):
    return build("youtube", "v3", developerKey=API_KEY)

def get_video_comments(service, video_id, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs, videoId=video_id).execute()

    while results:
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check if there are more comments to retrieve
        if "nextPageToken" in results:
            kwargs["pageToken"] = results["nextPageToken"]
            results = service.commentThreads().list(**kwargs, videoId=video_id).execute()
        else:
            break

    return comments

def save_comments_to_csv(comments, output_file):
    df = pd.DataFrame({"Comments": comments})
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Comments saved to '{output_file}'.")

def convert_video_to_frames(video_path, output_path='.'):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval for 1 frame per second
        frame_interval = int(fps)

        # Create output folder in the same directory as the video
        frames_folder = os.path.join(os.path.dirname(video_path), 'frames')
        os.makedirs(frames_folder, exist_ok=True)

        print(f"Converting video to frames (1 frame per second)...")
        print(f"FPS: {fps}, Total Frames: {frame_count}")

        # Read and save frames at 1 frame per second
        for frame_num in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as image file
            frame_filename = os.path.join(frames_folder, f"frame_{frame_num + 1:04d}.png")
            cv2.imwrite(frame_filename, frame)

        print(f"Frames saved to: {frames_folder}")
        separate_frames_with_faces(frames_folder)

        # Release the video capture object
        cap.release()

    except Exception as e:
        print(f"An error occurred: {e}")

def separate_frames_with_faces(frames_folder, output_folder='frames_with_faces'):
    try:
        # Create output folder for frames with faces
        output_path = os.path.join(os.path.dirname(frames_folder), output_folder)
        os.makedirs(output_path, exist_ok=True)

        # Load pre-trained Haarcascades classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print(f"Detecting faces in frames...")

        # Iterate through frames in the input folder
        for frame_filename in os.listdir(frames_folder):
            frame_path = os.path.join(frames_folder, frame_filename)

            # Read the frame
            frame = cv2.imread(frame_path)

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Save frames with faces to the output folder
            if len(faces) > 0:
                output_frame_path = os.path.join(output_path, frame_filename)
                cv2.imwrite(output_frame_path, frame)

        print(f"Frames with faces saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def comments_analysis(csv_file, model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    df = pd.read_csv(csv_file)

    predictions = model.predict(df)

    counts = pd.Series(predictions).value_counts()

    plt.bar(counts.index, counts.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(range(3), ['Negative', 'Neutral', 'Positive'])
    plt.title('Comments Sentiment Analysis')
    plt.show()

    return counts

        


if __name__== '__main__':
    app.run(debug=True)
