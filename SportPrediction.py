import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler


# Saving the model to a pickle file
model = pickle.load(open("bestmodel.pkl", "rb"))

# Loading the StandardScaler
scale = pickle.load(open("scale.pkl", "rb"))


# Defining the Streamlit app
def main():
    st.title("SportsPredictor")

    # Create sliders for the features
    st.write("Feature values for prediction:")
    movement_reactions = st.slider("movement_reactions", min_value=0, max_value=100, value=0)
    mentality_composure = st.slider("mentality_composure", min_value=0, max_value=100, value=0)
    passing = st.slider("passing", min_value=0, max_value=100, value=0)
    lcm = st.slider("lcm", min_value=0, max_value=100, value=0)
    cm = st.slider("cm", min_value=0, max_value=100, value=0)
    rcm = st.slider("rcm", min_value=0, max_value=100, value=0)
    potential = st.slider("potential", min_value=0, max_value=100, value=0)
    lm = st.slider("lm", min_value=0, max_value=100, value=0)
    rm = st.slider("rm", min_value=0, max_value=100, value=0)

  

    try:
        X= np.array([
            movement_reactions, mentality_composure, passing, lcm, cm, rcm, potential, lm, rm
        ]).astype(float).reshape(1, -1)


        if st.button("Predict Overall"):
            #Used the mean value and standard deviation value we got from our voting regressor to scale even though
            #we had loaded our scale
            mean_values = [65.75]  # mean value
            scale_values = [6.56] #standard deviation value
            scaled_input = (X - mean_values) / scale_values  # Apply the scaling
            prediction = model.predict(scaled_input)  # Making a prediction
            st.write(f"Overall Rating: {prediction[0]:.2f}")
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()







