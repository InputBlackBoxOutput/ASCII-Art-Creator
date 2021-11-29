import streamlit as st
from PIL import Image
import cv2
import numpy as np

import ascii_art

st.set_page_config(page_title='ASCII Art Creator', page_icon='🖼️', layout='centered')
st.image("images/banner.png", use_column_width=True)

st.image("images/sample-output-1.png", use_column_width=True)

st.markdown(
	'''
		#### How to use the website?
		1. Upload an image
		1. Set input image width
		1. Set the threshold such that optimal edges and thinned edges are observed
		1. Click on the 'Generate ASCII art' button and wait for a while

		#### How it works?
		1. Edges are detected from the image using dilation followed by subtraction with the original image
		1. Thinning operation is performed on the edges using Guo-Hall thinning algorithm
		1. Sub-images are obtained by using the sliding window technique and then passed to a CNN which determines the best character that represents the thinned edges present in the sub-image

	'''
	, unsafe_allow_html=True)

st.image("images/process.drawio.png", use_column_width=True)


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
	new_width = st.number_input("Input image width", min_value= 400, max_value=800, value=550, step=5)
	threshold_value = st.slider("Threshold", min_value=0, max_value=255, value=150, step=1)

	_, c1, c2, _ = st.columns([2, 4, 4, 2])
	edges = ascii_art.detect_edges(uploaded_image, threshold=threshold_value)
	thinned_edges = 255- 255 * ascii_art.thin_edges(edges)

	c1.image(255 * edges, use_column_width=True, caption="Edges")
	c2.image(thinned_edges, caption="Thinned edges")

	if st.button('Generate ASCII art'):
		with st.spinner('Processing the image. This may take a while'):
			artwork, artwork_text = ascii_art.generate_ascii_art(thinned_edges, new_width=new_width)

		st.image(artwork, use_column_width='always', caption="ASCII art")
		
		# formatted = ""
		# for each in artwork_text:
			# line = "".join(each) + '\r\n'
			# formatted += line

		# st.markdown(f'''<pre style="font-family: 'MS PGothic', 'Saitamaar', 'IPAMonaPGothic' !important;">{formatted}</pre>''', unsafe_allow_html=True)
		
		st.download_button(label="Download image", data=cv2.imencode('.jpg', artwork)[1].tobytes(), file_name="ascii-art.png", mime="image/png")

	st.markdown('<p style="text-align:center"> Press <kbd>Crtl</kbd> + <kbd>R</kbd> to reset </p>', unsafe_allow_html=True)

st.markdown('<hr> <h5>Made with lots of ⏱️, 📚 and ☕ by <a href="https://github.com/InputBlackBoxOutput">InputBlackBoxOutput</a><h5> <hr>', unsafe_allow_html=True)