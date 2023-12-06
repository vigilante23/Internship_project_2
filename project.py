import streamlit as st
from frames import load_video
from recognize_frame import predict
import os

if __name__ == "__main__":
    st.title("Action Recognition")
    video_path = r'C:\Users\HP\Documents\Internship_project_2\video\soccer.mp4'
    OUTPUT_PATH = r'C:\Users\HP\Documents\Internship_project_2\output_frames'
    filename = os.path.basename(video_path)
    video_filename, _ = os.path.splitext(filename)
    sample_video = load_video(video_path)
    predicted_action = predict(sample_video, video_filename,  OUTPUT_PATH)
