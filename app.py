import streamlit as st
from streamlit_option_menu import option_menu
from fastai.vision.all import *

import plotly.express as px
#---------------------------------------------------

st.set_page_config(
    page_title="UnlockAI", 
    page_icon='unlock.webp', 
    layout="centered"
)

hd_st_style = """
        <style>
         #MainMenu {visibility: hidden;}
         footer {visibility: hidden;}
         header {visibility: hidden;}
        </style>
        """

st.markdown(hd_st_style, unsafe_allow_html=True)

#-----------------------------------------------------------

MAIN_MENU = option_menu(
    menu_title=None,
    options=['Bosh sahifa', "Loyiha"],
    icons=['house', 'box'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding":"0!important", "background":"#fafafa", "margin":"0!important"},
        "nav-link-selected": {"background-color":"black"},
    }
)

#-----------------------------------------------------------------------------------------
def translate(text):
    d = {
        "Apple": "OLMA",
        "Lemon": "LIMON",
        "Grape": "UZUM",
        "Peach": "SHAFTOLI",
        "Banana": "BANAN"
    }

    return d.get(text)


def bosh_sahifa():
    st.subheader("Mevalarni klassifikatsiya qiluvchi model")
    st.image("lemon.gif")
    
def loyiha():
    file = st.file_uploader("Rasm yuklang", type=["jpg", 'png', 'jpeg'])
    #display and predict it
    if file:  
        st.image(file)
        image = PILImage.create(file)
        model = load_learner('fruits_model1.pkl')
            
        pred, pred_id, probs = model.predict(image)

        if probs[pred_id]*100 < 97.5 or pred == "Washing machine":
            st.warning("Bu rasm klassga oid emas!")
        else:
            st.success("Bu " + translate(pred))
            st.info(f"Aniqlilik: {probs[pred_id]*100:.1f}%")
            #plotly chart
            fig = px.bar(x=probs*100, y=model.dls.vocab)
            st.plotly_chart(fig)
            
        
#-------------------------------------------------------------------------------------

if MAIN_MENU == 'Bosh sahifa':
    bosh_sahifa()

if MAIN_MENU == 'Loyiha':
    loyiha()

#-------------------------------------------------------------------------

st.info("Model 5 xil turdagi mevalarni klassifikatsiya qila oladi. Bular: Olma, Shaftoli, Banan, Uzum, Limon")

st.text("UnlockAI??? 2022")