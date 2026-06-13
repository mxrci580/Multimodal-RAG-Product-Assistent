import streamlit as st
from app.multimodal_agent import MultimodalAgent

st.set_page_config(
    page_title="Multimodal Product Assistant",
    page_icon="🛍️"
)

st.title("🛍️ Multimodal Product Assistant")

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image_path = "temp_image.png"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(
        image_path,
        caption="Uploaded Image",
        width=300
    )

    if st.button("Find Similar Products"):

        with st.spinner("Analyzing image..."):

            agent = MultimodalAgent()

            response = agent.run(
                image_path
            )

        st.success("Done!")

        st.markdown(response)