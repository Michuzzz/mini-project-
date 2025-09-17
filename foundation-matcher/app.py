import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load dataset
df = pd.read_csv("foundation_dataset.csv")
df[['R', 'G', 'B']] = df[['R', 'G', 'B']].astype(int)

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Detect skin tone
def get_skin_tone(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None, None
    h, w, _ = image.shape
    skin_pixels = []
    for lm in results.multi_face_landmarks[0].landmark[234:454]:
        x, y = int(lm.x * w), int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            skin_pixels.append(image_rgb[y, x])
    if not skin_pixels:
        return None, None
    avg_color = np.mean(skin_pixels, axis=0).astype(int)
    return image, avg_color

# Recommend foundations
def recommend_foundation(avg_color):
    df['distance'] = np.sqrt(
        (df['R']-avg_color[0])**2 +
        (df['G']-avg_color[1])**2 +
        (df['B']-avg_color[2])**2
    )
    recommendations = df.sort_values(by="distance").head(3)
    return recommendations[['brand', 'product', 'name', 'in_india', 'R', 'G', 'B']]

# Apply foundation
def apply_foundation(image, foundation_rgb, mode="half"):
    overlay = image.copy()
    foundation_bgr = (foundation_rgb[2], foundation_rgb[1], foundation_rgb[0])
    foundation_layer = np.full_like(image, foundation_bgr, dtype=np.uint8)
    h, w, _ = image.shape
    if mode == "half":
        foundation_layer[:, :w//2, :] = image[:, :w//2, :]
    blended = cv2.addWeighted(foundation_layer, 0.5, image, 0.5, 0)
    return blended

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    shades_output = None
    error = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "⚠️ Please upload an image!"
            return render_template("index.html", error=error)

        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        image, avg_color = get_skin_tone(filepath)
        if image is None:
            error = "⚠️ Please upload a clear photo of a single human face without heavy makeup or shadows."
            return render_template("index.html", error=error)

        recommendations_df = recommend_foundation(avg_color)
        recommendations = recommendations_df[['brand','product','name','in_india']].to_dict(orient="records")

        # Generate half/full previews
        shades_output = []
        for i, row in recommendations_df.iterrows():
            shade_name = row['name'].replace(" ", "_")  # safe filename
            for mode in ["half", "full"]:
                out_path = f"static/uploads/{mode}_{i}_{shade_name}.jpg"
                applied = apply_foundation(image, [row['R'], row['G'], row['B']], mode)
                cv2.imwrite(out_path, applied)
                shades_output.append({
                    "img": out_path,
                    "label": f"{row['brand']} - {row['name']} ({mode})",
                    "shade_index": i
                })

    return render_template("index.html", recommendations=recommendations, shades_output=shades_output, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
