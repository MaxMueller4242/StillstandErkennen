import streamlit as st
import torch
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import csv

# Modell laden
@st.cache_resource
def load_model():
    model_path = "best (1).pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# Videoverarbeitung mit Fehlerbehandlung
def process_video(video_path, model, frame_step, confidence_threshold):
    st.write("\U0001F680 **Starte Videoverarbeitung...**")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Fehler: Das Video konnte nicht geöffnet werden!")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_process = max(1, total_frames // frame_step)

    st.write(f"🎥 **Gesamtzahl der Frames:** {total_frames}")
    st.write(f"⏱ **Framerate (FPS):** {fps}")
    st.write(f"⚡ **Geplante Verarbeitung:** {frames_to_process} Frames")

    progress_bar = st.progress(0)
    st_frame = st.empty()
    results_list = []
    frame_numbers = []
    class_names = []

    frame_index = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detections = detect_objects(frame_rgb, model, confidence_threshold)
            results_list.append(detections)

            for det in detections:
                frame_numbers.append(frame_index)
                class_names.append(det[0])

            st_frame.image(processed_frame, channels="RGB", caption=f"Frame {frame_index}/{total_frames}")
            processed_frames += 1
            progress = processed_frames / frames_to_process
            progress_bar.progress(min(progress, 1.0))

        frame_index += 1

    cap.release()
    plot_results(frame_numbers, class_names)
    plot_class_detections(frame_numbers, class_names)  # Neue Funktion aufrufen
    return results_list

# Bildverarbeitung mit YOLOv5
def detect_objects(image, model, confidence_threshold):
    results = model(image)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0]:
        if conf.item() >= confidence_threshold:
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = model.names[int(cls)]
            detections.append((class_name, x1, y1, x2, y2, conf.item()))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, detections

# Ergebnisse visualisieren
def plot_results(frame_numbers, class_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(frame_numbers, class_names, marker='o', color='blue', alpha=0.6)
    plt.xlabel("Frame Nummer")
    plt.ylabel("Klassennamen")
    plt.title("Erkannte Objekte über Frames hinweg")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)



def plot_class_detections(frame_numbers, class_names):
    plt.figure(figsize=(10, 2))  # Kleinere Höhe für den horizontalen Balken

    # Vektoren zum Speichern der Farben und der Spahn-Werte
    colors = []
    spahn_values = []

    # Erste Schleife: Abfrage der Klassen und Speichern der Farben und Spahn-Werte
    for class_name in class_names:
        if class_name == "WZ_Aufhanme_Dreht":
            color = 'green'
        elif class_name == "WZ_Aufhanme_Steht":
            color = 'orange'

        if class_name == "Spahn":
            spahn_values.append(1)
        else:
            spahn_values.append(0)

        colors.append(color)

    # Zusätzliche Schleife: Überprüfen des Wechsels der Klasse "Spahn" und Anpassung der Farben
    for i in range(1, len(spahn_values)):
        if spahn_values[i-1] == 1 and spahn_values[i] == 0 and colors[i] == 'orange':
            colors[i] = 'red'

    # Weitere Schleife: Vorwärts durchlaufen und 'orange' zu 'red' ändern
    for i in range(len(colors)):
        if colors[i] == 'red':
            for j in range(i + 1, len(colors)):
                if colors[j] == 'orange':
                    colors[j] = 'red'
                else:
                    break

    # Weitere Schleife: Rückwärts durchlaufen und 'orange' zu 'red' ändern
    for i in range(len(colors) - 1, -1, -1):
        if colors[i] == 'red':
            for j in range(i - 1, -1, -1):
                if colors[j] == 'orange':
                    colors[j] = 'red'
                else:
                    break

    # Zweite Schleife: Plotten der Balken
    for frame_number, color in zip(frame_numbers, colors):
        plt.barh(y=0, width=1, left=frame_number, color=color, alpha=0.6, edgecolor=color)

    plt.xlabel("Frame Nummer")
    plt.ylabel("Erkennung")
    plt.title("Klassendetections über Frames hinweg")
    plt.xticks(rotation=45)
    plt.yticks([])  # Entferne die y-Achse Ticks
    plt.grid(True, axis='x')

    # Legende hinzufügen
    green_patch = mpatches.Patch(color='green', label='Maschien in Arbeit')
    orange_patch = mpatches.Patch(color='orange', label='Maschien Steht')
    red_patch = mpatches.Patch(color='red', label='Stillstan wegen Spahn')
    plt.legend(handles=[green_patch, orange_patch, red_patch], loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    st.pyplot(plt)


# Funktion zum Speichern der Detektionen in einer CSV
def save_detections_to_csv(detections, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image/Frame", "Class", "X1", "Y1", "X2", "Y2", "Confidence"])
        for detection in detections:
            writer.writerow(detection)

# Streamlit UI
st.title("🔍 YOLOv5 Objekterkennung für Bilder & Videos")
uploaded_file = st.file_uploader("Lade ein Bild oder Video hoch", type=["jpg", "png", "jpeg", "mp4"])
frame_step = st.number_input("Nur jeden n-ten Frame analysieren", min_value=1, value=1, step=1)
confidence_threshold = st.slider("Mindestkonfidenz für Erkennung", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    model = load_model()

    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        processed_image, detections = detect_objects(image, model, confidence_threshold)
        st.image(processed_image, caption="Erkannte Objekte", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            output_csv = temp_file.name
            save_detections_to_csv(detections, output_csv)
            st.download_button("📥 CSV mit Detektionen herunterladen", data=open(output_csv, "rb").read(), file_name="detections.csv")

    elif uploaded_file.type == "video/mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.video(temp_video_path)
        results = process_video(temp_video_path, model, frame_step, confidence_threshold)

        video_detections = []
        for frame_index, frame_detections in enumerate(results):
            for det in frame_detections:
                video_detections.append([f"Frame {frame_index}", *det])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            output_csv = temp_file.name
            save_detections_to_csv(video_detections, output_csv)
            st.download_button("📥 CSV mit Detektionen herunterladen", data=open(output_csv, "rb").read(), file_name="detections.csv")

        os.remove(temp_video_path)
