import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os

# Path to the YOLOv5 model
model_path = 'best.pt'

# Load the YOLOv5 model
@st.cache_resource
def load_model(model_path):
    try:
        return torch.hub.load('.', 'custom', path=model_path, source='local')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise

model = load_model(model_path)

# Sidebar instructions
st.sidebar.title("How to Use")
st.sidebar.info("""
1.Upload a video.  
2.Wait for the app to process and track players and the ball.  
3.Wait for the processing video. 
4.Download the processed vedio.

""")

# Main app interface
st.title("🎾 TENNIS TRACKING 🎾")
st.markdown("#### To Detect and track players in your tennis videos.")

# Upload video file
uploaded_video = st.file_uploader("Upload your video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Show user that video is uploading
    st.write("Uploading video...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open the video for reading
    cap = cv2.VideoCapture(temp_video_path)

    # Prepare the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        output_video_path = temp_output.name

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = 0

    # Show processing message
    st.write("Processing video....")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)
        processed_frame = np.squeeze(results.render())

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame in the Streamlit app
        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Update progress bar and percentage
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        st.write(f"Processing... {int(progress * 100)}%")

    # Release video resources
    cap.release()
    out.release()

    # Notify user of completion
    st.success("✅ Processing complete! Download your video below.")

    # Add download button for processed video
    with open(output_video_path, 'rb') as file:
        st.download_button(
            label="⬇ Download Processed Video",
            data=file,
            file_name="processed_tennis_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
