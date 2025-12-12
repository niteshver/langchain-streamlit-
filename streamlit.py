import streamlit as st
st.title("Welcome to LLM Journey")
st.write('''
         Hi,What are u doing''')

select_box = st.selectbox("U like:", ["Ishita","Ur Ex","Both"])

st.success("You choice is good")

if st.button("Click me"):
    st.write("Ur senseis good")

check_box = st.checkbox("Do u like Ishita??")
if check_box:
    st.write("U are a true lover")


relation_ship = st.radio("Relationship StatusL",
                         ["Single","Committed","Bich Ka"])
st.info(f"U ar {relation_ship} now")
if relation_ship =="Single":
    st.balloons()
else:
    st.snow()

uploaded_file = st.file_uploader("Ypload ur pic")
if uploaded_file:
    st.image(uploaded_file)