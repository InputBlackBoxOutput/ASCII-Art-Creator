import cv2
import numpy as np
import streamlit as st

import model
import preprocess
import imutils

st.set_page_config(page_title='ASCII Art Creator', page_icon='🖼️', layout='centered')
st.image("images/banner.png", width='stretch')

st.image("images/sample-output-1.png", width='stretch')

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

st.image("images/process.drawio.png", width='stretch')

st.markdown("-----")
image_file = st.file_uploader("Upload an image file", type=['png', 'jpg'])

if image_file is not None:
    uploaded_image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

    _, c, _ = st.columns([2, 4, 2])
    c.image(uploaded_image, width='stretch', caption="Original Image")
    st.success("Image uploaded successfully!")

    st.markdown("-----")
    width = st.number_input("Input image width", min_value= 400, max_value=1000, value=640, step=10)
    threshold_value = st.slider("Threshold", min_value=0, max_value=255, value=150, step=1)

    _, c1, c2, _ = st.columns([2, 4, 4, 2])
    edges = preprocess.detect_edges(uploaded_image, threshold=threshold_value)
    thin = 255 - 255 * preprocess.thin_edges(edges)

    c1.image(255 * edges, width='stretch', caption="Edges")
    c2.image(thin, caption="Thinned edges")

    if st.button('Generate ASCII art', type="primary", width='stretch'):
        with st.spinner('Processing the image. This may take a while'):
            thin = imutils.resize(thin, width=width)
            thin = imutils.add_margin(thin)

            CNN = model.Model()
            artwork, text = CNN.generate(thin)

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
