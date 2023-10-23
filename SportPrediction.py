import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Saving the model to a pickle file
model = pickle.load(open("bestmodel.pkl", "rb"))

# Loading the StandardScaler
scale = pickle.load(open("scaled.pkl", "rb"))


# Defining the Streamlit app
def main():
    st.title("Sports Player Rating Predictor")

    # Creating sliders for the features
    st.write("Feature values for prediction:")
    movement_reactions = st.slider("movement_reactions", min_value=0, max_value=100, value=0)
    release_clause_eur = st.slider("release_clause_eur", min_value=0, max_value=100, value=0)
    age = st.slider("age", min_value=0, max_value=100, value=0)
    wage_eur = st.slider("wage_eur", min_value=0, max_value=100, value=0)
    potential = st.slider("potential", min_value=0, max_value=100, value=0)
    value_eur= st.slider("value_eur", min_value=0, max_value=100, value=0)
    gk = st.slider("gk", min_value=0, max_value=100, value=0)
    

    try:
        X= np.array([
            movement_reactions,release_clause_eur,age,wage_eur, potential,value_eur,gk
        ]).astype(float).reshape(1, -1)
##We are using a computation to get an appropriate scale from our main model on colab using the
##mean and standard deviation of our best model, Voting Regression. This mean and standard deviation
#was generated using code on our best model.
        scaled_input=(X-65.51611947627754)/6.980938569708749
        if scaled_input.dtype!=np.float64:
            st.write("Wrong dzata type")


        if st.button("Predict Overall Rating"):
            prediction = model.predict(scaled_input)  # Making a prediction
            st.write(f"Overall Rating: {prediction[0]:.2f}")


            # Calculating the standard deviation and mean of the predictions
            std_predictions = prediction.std()
            mean_predictions = prediction.mean()


            # Finding the coefficient of variation(CV) as a confidence score 
            confidence_score= std_predictions /mean_predictions




    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()







