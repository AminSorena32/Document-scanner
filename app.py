import io

import streamlit as st
import cv2 as cv
from PIL import Image

from scan.scanner_module import process_and_scan
from scan.improvement_module import Improvement

st.title("Scanner App")

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Camera")
selected_file = uploaded_file or camera_file

show_debug = st.checkbox("Show debug/visualization pipeline")

if selected_file:
    if st.button("Scan"):
        try:
            if show_debug:
                scanned_img, scan_steps = process_and_scan(selected_file, debug=True)
            else:
                scanned_img = process_and_scan(selected_file)

            st.subheader("Scanned Image")
            scanned_img_rgb = cv.cvtColor(scanned_img, cv.COLOR_BGR2RGB)
            st.image(scanned_img_rgb, caption="Scanned Result")

            if show_debug:
                improved_img, improve_steps = Improvement(scanned_img, debug=True)
            else:
                improved_img = Improvement(scanned_img)

            st.subheader("Improved Image")
            st.image(improved_img, caption="Improved Result", channels="GRAY")

            if show_debug:
                with st.expander("🔍 Debug: Detection & Enhancement Pipeline", expanded=True):
                    method = scan_steps.pop("_corner_detection_method")
                    st.caption(f"Corner detection method used: **{method}**")

                    st.markdown("**Document Detection**")
                    cols = st.columns(2)
                    for i, (label, img) in enumerate(scan_steps.items()):
                        with cols[i % 2]:
                            if img.ndim == 2:
                                st.image(img, caption=label, channels="GRAY")
                            else:
                                st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption=label)

                    st.markdown("**Image Enhancement**")
                    cols2 = st.columns(2)
                    for i, (label, img) in enumerate(improve_steps.items()):
                        with cols2[i % 2]:
                            st.image(img, caption=label, channels="GRAY")

            st.subheader("Download")
            col_png, col_pdf = st.columns(2)

            ok, buffer = cv.imencode(".png", improved_img)
            if not ok:
                st.error("Failed to encode image.")
            else:
                with col_png:
                    st.download_button(
                        label="Download as PNG",
                        data=buffer.tobytes(),
                        file_name="scanned_output.png",
                        mime="image/png"
                    )

            # PDF export: wrap the improved (grayscale) image into a single-page PDF
            pdf_image = Image.fromarray(improved_img).convert("RGB")
            pdf_buffer = io.BytesIO()
            pdf_image.save(pdf_buffer, format="PDF")
            pdf_buffer.seek(0)

            with col_pdf:
                st.download_button(
                    label="Download as PDF",
                    data=pdf_buffer,
                    file_name="scanned_output.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")
