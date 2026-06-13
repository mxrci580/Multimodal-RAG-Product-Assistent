import streamlit as st
from app.multimodal_agent import MultimodalAgent

st.set_page_config(
    page_title="Multimodal Product Assistant",
    page_icon="🛍️"
)

st.title("🛍️ Multimodal Product Assistant")

st.write(
    "Upload a product image and get recommendations."
)

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["png", "jpg", "jpeg"],
    key="product_image"
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

    if st.button(
        "Find Similar Products",
        key="find_products"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            agent = MultimodalAgent()

            response = agent.run(
                image_path
            )

        st.success("Analysis Complete!")

        st.markdown(
            response["answer"]
        )

        products = response["products"]

        st.subheader(
            "Top Matching Products"
        )

        for product in products:

            col1, col2 = st.columns(
                [1, 2]
            )

            with col1:

                st.image(
                    product["image_url"],
                    width=150
                )

            with col2:

                st.markdown(
                    f"### {product['product_name']}"
                )

                st.write(
                    f"💰 Price: {product['price']}"
                )

                st.write(
                    f"⭐ Rating: {product['rating']}"
                )

            st.divider()