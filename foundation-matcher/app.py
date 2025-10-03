import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)
# Set the upload folder to an absolute path
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset - use absolute path
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foundation_dataset.csv')
df = pd.read_csv(dataset_path)
df[['R', 'G', 'B']] = df[['R', 'G', 'B']].astype(int)

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    try:
        img = Image.open(io.BytesIO(file_stream.read()))
        img.verify()
        file_stream.seek(0)
        return True
    except:
        return False

def detect_undertone(rgb):
    r, g, b = rgb
    warmth = (r + g) / 2 - b
    if warmth > 10:
        return "Warm"
    elif warmth < -10:
        return "Cool"
    else:
        return "Neutral"

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Detect skin tone
def get_skin_tone(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "Could not read the image."
    
    # Resize large images for faster processing
    max_dimension = 800
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None, None, "⚠️ No face detected. Please upload a clear human face."
    if len(results.multi_face_landmarks) > 1:
        return None, None, "⚠️ Multiple faces detected. Please upload only one face."
    
    h, w, _ = image.shape
    skin_pixels = []
    
    # Use more comprehensive landmarks for skin detection
    skin_landmarks = list(range(1, 11)) + list(range(151, 161)) + list(range(234, 454))
    
    for lm_idx in skin_landmarks:
        lm = results.multi_face_landmarks[0].landmark[lm_idx]
        x, y = int(lm.x * w), int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            skin_pixels.append(image_rgb[y, x])
    
    if not skin_pixels:
        return None, None, "⚠️ Could not detect skin properly. Avoid heavy makeup or shadows."
    
    avg_color = np.mean(skin_pixels, axis=0).astype(int)
    return image, avg_color, None

# Recommend foundations
def recommend_foundation(avg_color, top_n=3):
    df['distance'] = np.sqrt(
        (df['R']-avg_color[0])**2 +
        (df['G']-avg_color[1])**2 +
        (df['B']-avg_color[2])**2
    )
    recommendations = df.sort_values(by="distance").head(top_n)
    return recommendations

# Apply foundation to image (improved to avoid hair, eyes, teeth)
def apply_foundation(image, foundation_rgb, mode="half"):
    # Create a mask for the face area
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w, _ = image.shape
    
    # Get face landmarks
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return image
    
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    # Define face contour (skin area)
    face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_points = [points[i] for i in face_contour if i < len(points)]
    
    # Define areas to exclude (eyes, mouth, eyebrows)
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    mouth = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
    
    # Draw face contour
    cv2.fillConvexPoly(mask, np.array(face_points), 255)
    
    # Exclude eyes and mouth
    for exclude_region in [left_eye, right_eye, mouth]:
        exclude_points = [points[i] for i in exclude_region if i < len(points)]
        cv2.fillConvexPoly(mask, np.array(exclude_points), 0)
    
    # Apply Gaussian blur to the mask for smoother edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Create foundation layer
    foundation_bgr = (foundation_rgb[2], foundation_rgb[1], foundation_rgb[0])
    foundation_layer = np.full_like(image, foundation_bgr)
    
    # Blend the foundation layer with the original image using the mask
    blended = image.copy()
    for i in range(3):
        blended[:,:,i] = image[:,:,i] * (1 - mask/255.0) + foundation_layer[:,:,i] * (mask/255.0)
    
    # If mode is half, only apply to the left half
    if mode == "half":
        half_mask = np.zeros_like(mask)
        half_mask[:, :w//2] = 1
        mask = mask * half_mask
        for i in range(3):
            blended[:,:,i] = image[:,:,i] * (1 - mask/255.0) + foundation_layer[:,:,i] * (mask/255.0)
    
    return blended

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    shades_output = None
    error = None
    uploaded_file = None
    skin_tone = None
    skin_tone_hex = None
    undertone = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "⚠️ Please upload an image!"
            return render_template("index.html", error=error)

        file = request.files["file"]
        if not allowed_file(file.filename):
            error = "⚠️ Invalid file type! Only jpg, jpeg, png allowed."
            return render_template("index.html", error=error)
        
        # Validate image
        if not validate_image(file.stream):
            error = "⚠️ Invalid image file! Please upload a valid image."
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        uploaded_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file)

        image, avg_color, face_error = get_skin_tone(uploaded_file)
        if face_error:
            error = face_error
            return render_template("index.html", error=error)

        # Get skin tone info
        skin_tone = avg_color.tolist()
        skin_tone_hex = rgb_to_hex(skin_tone)
        undertone = detect_undertone(skin_tone)

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

    return render_template("index.html", 
                           recommendations=recommendations, 
                           shades_output=shades_output, 
                           error=error, 
                           uploaded_file=uploaded_file,
                           skin_tone=skin_tone,
                           skin_tone_hex=skin_tone_hex,
                           undertone=undertone)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
