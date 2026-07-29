import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib


st.set_page_config(
    page_title="E-Commerce Recommendation",
    page_icon="🛒",
    layout="wide"
)


# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp {
    background-color: #000000;
}


h1 {
    color: #8B0000;
    text-align:center;
    font-size:45px;
}


h2,h3 {
    color:#B22222;
}


p, label {
    color:white !important;
}


.stButton button {

    background-color:#8B0000;
    color:white;
    width:100%;
    height:45px;
    border-radius:10px;
    font-size:18px;

}


.stButton button:hover {

    background-color:#FF0000;
    color:white;

}


div[data-testid="stSuccess"] {

    background-color:#111111;
    border-left:5px solid #8B0000;

}


div[data-testid="stError"] {

    background-color:#111111;
    border-left:5px solid red;

}


</style>

""", unsafe_allow_html=True)



# ================= LOAD FILES =================


model = tf.keras.models.load_model(
    "recommendation_model.keras"
)


scaler = joblib.load(
    "scaler.pkl"
)


label_encoders = joblib.load(
    "label_encoders.pkl"
)


df = pd.read_csv(
    "Ecommerce_Personalized_Recommendation_Dataset.csv"
)



# ================= SIDEBAR =================


with st.sidebar:


    st.title("🛒 Project Info")


    st.write(
    """
    **Project**

    E-Commerce Product Recommendation System


    **Model**

    Artificial Neural Network (ANN)


    **Developer**

    Dhaniya J U


    **Department**

    Artificial Intelligence
    and Data Science


    **Technologies**

    Python

    TensorFlow

    Keras

    Scikit-Learn

    Streamlit
    """
    )



# ================= TITLE =================


st.title(
    "🛒 E-Commerce Product Recommendation System"
)


st.markdown(
"""
<h3 style='text-align:center'>
ANN Based Recommendation System
</h3>
""",
unsafe_allow_html=True
)


st.markdown(
"""
<p style='text-align:center'>
AI Powered Personalized Product Recommendation
</p>
""",
unsafe_allow_html=True
)


st.markdown("---")



# ================= INPUT SECTION =================


st.header("📝 Customer & Product Details")



col1,col2 = st.columns(2)



with col1:


    st.subheader("👤 Customer Details")


    category = st.selectbox(
        "Category",
        df["Category"].unique()
    )


    brand = st.selectbox(
        "Brand",
        df["Brand"].unique()
    )


    age = st.number_input(
        "User Age",
        18,
        80,
        25
    )


    gender = st.selectbox(
        "Gender",
        df["User_Gender"].unique()
    )


    location = st.selectbox(
        "Location",
        df["User_Location"].unique()
    )


    device = st.selectbox(
        "Device Type",
        df["Device_Type"].unique()
    )




with col2:


    st.subheader("🛍 Shopping Behaviour")


    session = st.number_input(
        "Session Duration",
        1,
        300,
        30
    )


    pages = st.number_input(
        "Pages Viewed",
        1,
        100,
        10
    )


    time = st.selectbox(
        "Time Of Day",
        df["Time_of_Day"].unique()
    )


    purchases = st.number_input(
        "Previous Purchases",
        0,
        50,
        5
    )


    rating = st.slider(
        "User Rating",
        1.0,
        5.0,
        4.0
    )


    price = st.number_input(
        "Product Price",
        1.0,
        100000.0,
        500.0
    )


    discount = st.selectbox(
        "Discount Applied",
        [0,1]
    )



st.markdown("---")



# ================= PREDICTION =================


if st.button(
    "🚀 Recommend Product"
):


    input_data = pd.DataFrame({

        "Category":[category],

        "Brand":[brand],

        "User_Age":[age],

        "User_Gender":[gender],

        "User_Location":[location],

        "Device_Type":[device],

        "Session_Duration_Min":[session],

        "Pages_Viewed":[pages],

        "Time_of_Day":[time],

        "Previous_Purchases":[purchases],

        "User_Rating":[rating],

        "Product_Price":[price],

        "Discount_Applied":[discount],


        "Graph_Similarity_Score":[0.8],

        "Federated_Cluster_ID":[1],

        "Local_Model_Accuracy":[0.95],

        "Global_Model_Weight":[0.9],

        "Personalization_Factor":[0.8],

        "Purchase_Probability":[0.7]

    })



    categorical_columns = [

        "Category",

        "Brand",

        "User_Gender",

        "User_Location",

        "Device_Type",

        "Time_of_Day"

    ]



    # Encoding

    for col in categorical_columns:


        input_data[col] = label_encoders[col].transform(
            input_data[col].astype(str)
        )



    input_data = input_data.astype(float)



    # Scaling

    input_data = scaler.transform(
        input_data
    )



    # Prediction

    prediction = model.predict(
        input_data
    )


    probability = float(
        prediction[0][0]
    )



    score = probability * 100



    st.markdown("---")


    st.subheader(
        "🤖 Recommendation Result"
    )



    st.metric(
        "Recommendation Probability",
        f"{score:.2f}%"
    )


    st.progress(
        probability
    )



    if probability >= 0.5:


        st.success(
            "✅ Product Recommended"
        )


        st.info(
            f"""
            Recommended Category: {category}

            Recommended Brand: {brand}

            Product Price: ₹{price}
            """
        )


    else:


        st.error(
            "❌ Product Not Recommended"
        )