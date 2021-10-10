import streamlit as st
from PIL import Image
import cv2
import numpy as np

import ascii_art

st.set_page_config(page_title='ASCII Art Creator', page_icon='🖼️', layout='centered')
st.image("images/banner.png", use_column_width=True)

st.markdown(
	'''
		#### Generate ASCII art using computer vision
		<br>
	'''
	, unsafe_allow_html=True)

st.image("images/sample-output-1.png", use_column_width=True)
st.image("images/sample-output-2.png", use_column_width=True)

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("-----")
image_file = st.file_uploader("Upload an image file", type=['png', 'jpg'])

if image_file is not None:
	uploaded_image = Image.open(image_file)
	uploaded_image = np.array(uploaded_image)

	_, c, _ = st.columns([2,4,2])
	c.image(image_file, use_column_width=True, caption='Original Image')
	st.success("Image uploaded successfully!")

	st.markdown("-----")
	new_width = st.number_input("Output image width", min_value= 400, max_value=1000, value=550, step=5)
	threshold_value = st.slider("Threshold for binary thresholding", min_value=0, max_value=255, value=150, step=1)

	_, c1, c2, _ = st.columns([2, 4, 4, 2])
	edges = ascii_art.detect_edges(uploaded_image, threshold=threshold_value)
	thinned_edges = 255- 255 * ascii_art.thin_edges(edges)

	c1.image(255 * edges, use_column_width=True, caption="Edges")
	c2.image(thinned_edges, caption="Thinned edges")

	if st.button('Generate ASCII art'):
		with st.spinner('Processing the image. This may take a while'):
			artwork = ascii_art.generate_ascii_art(thinned_edges, new_width=new_width)
		st.image(artwork, use_column_width='always', caption="ASCII art")		
		st.download_button(label="Download image", data=cv2.imencode('.jpg', artwork)[1].tobytes(), file_name="ascii-art.png", mime="image/png")

	st.markdown('<p style="text-align:center"> Press <kbd>Crtl</kbd> + <kbd>R</kbd> to reset the application </p>', unsafe_allow_html=True)

st.markdown('<hr> <h5>Made with lots of ⏱️, 📚 and ☕ by <a href="https://github.com/InputBlackBoxOutput">InputBlackBoxOutput</a><h5> <hr>', unsafe_allow_html=True)