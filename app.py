import streamlit as st
import cv2 as cv

from scan.scanner_module import process_and_scan
from scan.improvement_module import Improvement

st.title("Scanner App")

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Camera")

selected_file = uploaded_file or camera_file

if selected_file:
    if st.button("Scan"):
        try:
            scanned_img = process_and_scan(selected_file)

            st.subheader("Scanned Image")
            scanned_img_rgb = cv.cvtColor(scanned_img, cv.COLOR_BGR2RGB)
            st.image(scanned_img_rgb, caption="Scanned Result")

            improved_img = Improvement(scanned_img)

            st.subheader("Improved Image")
            st.image(improved_img, caption="Improved Result", channels="GRAY")
            
            ok, buffer = cv.imencode(".png", improved_img)
            if not ok:
                st.error("Failed to encode image.")
            else:
                st.download_button(
                    label="Download Final Image",
                    data=buffer.tobytes(),
                    file_name="scanned_output.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Error: {e}")

