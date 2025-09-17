import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

# Load foundation dataset
df = pd.read_csv("foundation_dataset.csv.csv")

# Function to get dominant skin tone color from image
def get_dominant_color(image):
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=1, random_state=42).fit(img_array)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

# Function to recommend foundation shades
def recommend_foundations(image):
    if image is None:
        return "Please upload a valid image.", None

    # Extract dominant color from uploaded face image
    dominant_color = get_dominant_color(image)

    # Compare with dataset RGB values
    df["distance"] = np.sqrt(
        (df["R"] - dominant_color[0])**2 +
        (df["G"] - dominant_color[1])**2 +
        (df["B"] - dominant_color[2])**2
    )

    # Get top 3 matches
    top_matches = df.sort_values("distance").head(3)

    results = []
    color_swatches = []

    for _, row in top_matches.iterrows():
        results.append(f"**Brand:** {row['brand']}\n**Product:** {row['product']}\n**Shade:** {row['name']}")
        color_swatches.append(row["hex"])

    return "\n\n".join(results), color_swatches


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(" Your PerfectHue ")
    gr.Markdown("Upload a clear face image (natural light, no heavy makeup). we will recommend the **Top 3 foundation shades** closest to your skin tone.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload your face image")
        output_text = gr.Textbox(label="Top 3 Recommendations", lines=10)
        output_colors = gr.ColorPicker(label="Shade Previews", interactive=False)

    submit_btn = gr.Button("Find My Match")
    submit_btn.click(fn=recommend_foundations, inputs=image_input, outputs=[output_text, output_colors])

demo.launch()
