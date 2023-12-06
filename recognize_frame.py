from urllib import request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os

KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]

i3d_model_kinetics = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def predict(sample_video, video_filename, output_path):
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
    logits = i3d_model_kinetics(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    top_action_index = np.argmax(probabilities)
    top_action = labels[top_action_index]
    print(f"Top Action: {top_action} ({probabilities[top_action_index] * 100:.2f}%)")
    first_frame = sample_video[0]
    frame_with_text = (first_frame * 255.0).astype(np.uint8)
    frame_with_text_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(f"Action: {top_action}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_position = (10, 30 + text_size[1])  
    rectangle_height = text_size[1] + 10
    cv2.rectangle(frame_with_text_bgr, (10, 10), (10 + text_size[0] + 10, 10 + rectangle_height), (0, 0, 0), -1)
    text_color = (255, 0, 0) 
    cv2.putText(frame_with_text_bgr, f"Action: {top_action}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    resized_frame = cv2.resize(frame_with_text_bgr, (500, 500))
    output_frame_path = os.path.join(output_path, f"{video_filename}_annotated.jpg")
    cv2.imwrite(output_frame_path, resized_frame)
    return top_action
