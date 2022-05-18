import os
import streamlit as st
import gdown
from packaging.version import Version

from infer_func import convert

ROOT = os.path.dirname(os.path.abspath(__file__))

EXAMPLES = {
    'content': {
        'Brad Pitt': ROOT + '/examples/content/brad_pitt.jpg'
    },
    'style': {
        'Flower of Life': ROOT + '/examples/style/flower_of_life.jpg'
    }
}

VGG_WEIGHT_URL = 'https://drive.google.com/uc?id=1UcSl-Zn3byEmn15NIPXMf9zaGCKc2gfx'
DECODER_WEIGHT_URL = 'https://drive.google.com/uc?id=18JpLtMOapA-vwBz-LRomyTl24A9GwhTF'

VGG_WEIGHT_FILENAME = ROOT + '/vgg.pth'
DECODER_WEIGHT_FILENAME = ROOT + '/decoder.pth'


@st.cache
def download_models():
    with st.spinner(text="Downloading VGG weights..."):
        gdown.download(VGG_WEIGHT_URL, output=VGG_WEIGHT_FILENAME)
    with st.spinner(text="Downloading Decoder weights..."):
        gdown.download(DECODER_WEIGHT_URL, output=DECODER_WEIGHT_FILENAME)


def image_getter(image_kind):

    image = None

    options = ['Use Example Image', 'Upload Image']

    if Version(st.__version__) >= Version('1.4.0'):
        options.append('Open Camera')

    option = st.selectbox(
        'Choose Image',
        options, key=image_kind)

    if option == 'Use Example Image':
        image_key = st.selectbox(
            'Choose from examples',
            EXAMPLES[image_kind], key=image_kind)
        image = EXAMPLES[image_kind][image_key]

    elif option == 'Upload Image':
        image = st.file_uploader("Upload an image", type=['png', 'jpg', 'PNG', 'JPG', 'JPEG'], key=image_kind)
    elif option == 'Open Camera':
        image = st.camera_input('', key=image_kind)

    return image


if __name__ == '__main__':

    st.set_page_config(layout="wide")
    st.header('Adaptive Instance Normalization demo based on '
              '[2022-AdaIN-pytorch](https://github.com/media-comp/2022-AdaIN-pytorch)')

    download_models()
    # col1, col2, col3, col4 = st.columns((2, 2, 1, 3))
    col1, col2, col3 = st.columns((3, 4, 4))
    with col1:
        st.subheader('Content Image')
        content = image_getter('content')
        st.subheader('Style Image')
        style = image_getter('style')
    with col2:
        img1 = content if content is not None else 'examples/img.png'
        img2 = style if style is not None else 'examples/img.png'
        if img1 is not None:
            st.image(img1, width=None, caption='Content Image')
        if img2 is not None:
            st.image(img2, width=None, caption='Style Image')

    with col3:
        color_control = st.checkbox('Preserve content image color')
        alpha = st.slider('Strength of style transfer', 0.0, 1.0, 1.0, 0.01)
        process = st.button('Stylize')

    if content is not None and style is not None and process:
        print(content, style)
        with col3:
            with st.spinner('Processing...'):
                output_image = convert(content, style, VGG_WEIGHT_FILENAME, DECODER_WEIGHT_FILENAME, alpha, color_control)

            st.image(output_image, width=None, caption='Stylized Image')

