import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import io
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
import os

st.set_page_config(page_title="Paint by Numbers Generator", layout="centered")
st.title("🎨 Paint by Numbers Generator")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

num_colors = st.slider("Number of Colors", 2, 20, 8)
brush_size = st.slider("Brush Size", 1, 10, 1)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_array = np.array(img.resize((300, 300)))
    st.image(img_array, caption="Original Image", use_column_width=True)

    img_flat = img_array.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(img_flat)
    labels = kmeans.labels_.reshape(img_array.shape[:2])
    centers = kmeans.cluster_centers_.astype("uint8")
    simplified = centers[labels]

    outlines = np.ones(labels.shape, dtype=np.uint8) * 255
    for i in range(num_colors):
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outlines, contours, -1, 0, brush_size)

    numbers = outlines.copy()
    for i in range(num_colors):
        mask = (labels == i).astype(np.uint8)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(numbers, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    def to_img_bytes(arr, convert_rgb=False):
        if convert_rgb:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)
        return buf

    color_buf = to_img_bytes(simplified, convert_rgb=True)
    outline_buf = to_img_bytes(outlines)
    number_buf = to_img_bytes(numbers)

    st.subheader("Downloads")
    st.download_button("🖼️ Download Simplified Image", color_buf, file_name="color.png")
    st.download_button("🧾 Download Outline", outline_buf, file_name="outline.png")
    st.download_button("🔢 Download Numbered Image", number_buf, file_name="numbered.png")

    legend = [(i + 1, tuple(map(int, rgb))) for i, rgb in enumerate(centers)]

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Paint by Numbers")
    y -= 20

    for label, buf in zip(["Simplified", "Outline", "Numbered"], [color_buf, outline_buf, number_buf]):
        img = Image.open(buf)
        img.thumbnail((500, 300))
        c.setFont("Helvetica", 12)
        c.drawString(margin, y, label)
        y -= img.height + 10
        c.drawImage(ImageReader(img), margin, y)
        y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Legend:")
    y -= 20

    for idx, rgb in legend:
        if y < 50:
            c.showPage()
            y = height - margin
        c.setFillColor(Color(rgb[0]/255, rgb[1]/255, rgb[2]/255))
        c.rect(margin, y - 10, 10, 10, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(margin + 15, y - 2, f"{idx}: RGB {rgb}")
        y -= 15

    c.save()
    pdf_buf.seek(0)
    st.download_button("📄 Download PDF", pdf_buf, file_name="paint_by_numbers.pdf")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        zf.writestr("color.png", color_buf.getvalue())
        zf.writestr("outline.png", outline_buf.getvalue())
        zf.writestr("numbered.png", number_buf.getvalue())
        zf.writestr("legend.txt", '\n'.join([f"{idx}: RGB {rgb}" for idx, rgb in legend]))
    zip_buf.seek(0)

    st.download_button("🗜️ Download All as ZIP", zip_buf, file_name="paint_by_numbers_output.zip")
