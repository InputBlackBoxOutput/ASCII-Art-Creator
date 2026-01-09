import cv2
import numpy as np
from PIL import Image
import streamlit as st

import preprocess
import model
import imutils

st.set_page_config(page_title='ASCII Art Creator', page_icon='🖼️', layout='centered')
st.image("images/banner.png", width='stretch')

# st.image("images/.png", width='stretch')

st.markdown(
	"\n".join([
		"#### How to use the website?",
		"1. Upload an image",
		"1. Set input image width", 
		"1. Set the threshold such that optimal edges and thinned edges are observed",
		"1. Click on the 'Generate ASCII art' button and wait for a while",
		"",
		"#### How it works?",
		"1. Edges are detected from the image using dilation followed by subtraction with the original image",
		"1. Thinning operation is performed on the edges using Guo-Hall thinning algorithm",
		"1. Sub-images are obtained by using the sliding window technique and then passed to a CNN which determines the best character that represents the thinned edges present in the sub-image"
	])
	, unsafe_allow_html=True)

st.image("images/process.drawio.svg", width='stretch')

st.text("Made with lots of ⏱️, 📚 and ☕ by InputBlackBoxOutput")

st.markdown("-----")


options = {
    0: "Cartoon image with black outline [Recommended]",
    1: "Cartoon image with with less vibrant colour segments [Not implemented yet!]"
}
img_type = st.selectbox("Select image type:", options.values())

img_file = st.file_uploader("Upload a cartoon image file", type=['png', 'jpg'])
if img_file is not None:
    img = np.array(Image.open(img_file))

    _, c, _ = st.columns([2, 4, 2])
    c.image(img, width='stretch', caption="Original Image")
    st.success("Image uploaded successfully!")

    st.markdown("-----")
    width = st.number_input("Approximate output image width", min_value= 400, max_value=1000, value=640, step=10)
    deblur = st.number_input("Deblur intensity", min_value=0, max_value=3, value=0, step=1)

    if img_type == options[0]:
        threshold = st.slider("Edge detection threshold", min_value=0, max_value=255, value=150, step=10)
        img = preprocess.deblur(img, intensity=deblur)
        img = preprocess.detect_edges(img, threshold=threshold)
        st.image(img, width="stretch", caption="Outline")

    else:
        img = preprocess.deblur(img, intensity=deblur)
        img = preprocess.detect_edges_dnn(img)
        st.image(img, caption="Detected edges", width='stretch')

    if st.button('Generate ASCII art', type="primary", width='stretch'):
        with st.spinner('Processing the image. This may take a while'):
            img = imutils.resize(img, width=width)
            img = imutils.margin(img)

            CNN = model.Model()
            artwork, text = CNN.generate(img)

        st.image(artwork, width="stretch", caption="ASCII art")

        st.download_button(
			label="Download image",
			data=cv2.imencode('.jpg', artwork)[1].tobytes(),
			file_name="ascii-art.png",
			mime="image/png",
			width='stretch'
		)


