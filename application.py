import cv2
from cv2.gapi import imgproc
import numpy as np
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

st.markdown("-----")
image_file = st.file_uploader("Upload a cartoon image file", type=['png', 'jpg'])

if image_file is not None:
    uploaded_image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    _, c, _ = st.columns([2, 4, 2])
    c.image(uploaded_image, width='stretch', caption="Original Image")
    st.success("Image uploaded successfully!")

    st.markdown("-----")
    width = st.number_input("Approximate output image width", min_value= 400, max_value=1000, value=640, step=10)
    deblur = st.number_input("Deblur intensity", min_value=0, max_value=3, value=0, step=1)
    threshold = st.slider("Edge detection threshold", min_value=0, max_value=255, value=150, step=1)

    _, c1, c2, _ = st.columns([2, 4, 4, 2])
    img = preprocess.deblur(uploaded_image, intensity=deblur)
    img = preprocess.detect_edges(img, threshold=threshold)
    c1.image(255 * img, width='stretch', caption="Edges")

    img = 255 - 255 * preprocess.thin_edges(img)
    c2.image(img, caption="Thinned edges")

    if st.button('Generate ASCII art', type="primary", width='stretch'):
        with st.spinner('Processing the image. This may take a while'):
            img = imutils.resize(img, width=width)
            img = imutils.margin(img)

            CNN = model.Model()
            artwork, text = CNN.generate(img)

        st.image(artwork, width="stretch", caption="ASCII art")

        # Download text
        # formatted = ""
        # for each in artwork_text:
        # line = "".join(each) + '\r\n'
        # formatted += line

        # st.markdown(f'''<pre style="font-family: 'MS PGothic', 'Saitamaar', 'IPAMonaPGothic' !important;">{formatted}</pre>''', unsafe_allow_html=True)

        st.download_button(
			label="Download image", 
			data=cv2.imencode('.jpg', artwork)[1].tobytes(), 
			file_name="ascii-art.png", 
			mime="image/png",
			width='stretch'
		)

    st.markdown(
        '<p style="text-align:center"> Press <kbd>Crtl</kbd> + <kbd>R</kbd> to reset </p>', 
        unsafe_allow_html=True
    )

st.markdown(
    "\n".join(
        [
            "---",
            "##### Made with lots of ⏱️, 📚 and ☕ by [InputBlackBoxOutput](https://github.com/InputBlackBoxOutput)",
        ]
    )
)
