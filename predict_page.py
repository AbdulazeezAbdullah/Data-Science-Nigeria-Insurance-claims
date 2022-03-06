import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('Random_Forest_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

data = load_model()

model = data["model"]

incident_type_encode = data["incident_type_encoder"]
collision_type_encode = data["collision_type_encoder"]
incident_severity_encode = data["incident_severity_encoder"]
authorities_contacted_encode = data["authorities_contacted_encoder"]
incident_state_encode = data["incident_state_encoder"]
auto_model_encode = data["auto_model_encoder"]

def show_predict_page():
    st.header("Insurance Claim Prediction")

    st.subheader("We need some informatuion to predict the total amount to be claimed by the insured")
    
    authorities = (
                        'Ambulance',
                        'Fire',
                        'None',
                        'Other',
                        'Police'  
    )
    
    auto_model = (
                    '93','95','3 Series','92x','A3','A5','Accord','C300','Camry','Civic','Corolla','CRV','E400','Escape','F150','Forrestor','Fusion','Grand Cherokee',
                    'Highlander','Impreza','Jetta','Legacy','M5','Malibu','Maxima','MDX','ML350','Neon','Passat','Pathfinder','RAM','RSX','Silverado','Tahoe','TL','Ultima',
                    'Wrangler','X5','X6'
    )
    
    collision_type = (
                        'Not answered',
                        'Front Collision',
                        'Rear Collision',
                        'Side Collision'
    )
    
    incident_severity = (
                            'Major Damage',
                            'Minor Damage',
                            'Total Loss',
                            'Trivial Damage'
        
    )
    
    incident_state = ('NC','NY','OH','PA','SC','VA','WV')
    
    incident_type = (
                        'Multi-vehicle Collision',
                        'Parked Car',
                        'Single Vehicle Collision',
                        'Vehicle Theft'
    )
    
    Age = st.slider('Age', min_value=1, max_value=70, step=1)
    Authorities = st.selectbox('Authorities Contacted', authorities)
    Auto_Model = st.selectbox('Car Model', auto_model)
    Collision_type = st.selectbox('Type of collision', collision_type)
    Incident_severity = st.selectbox('Severity of the incident', incident_severity)
    Incident_state = st.selectbox('State Where incident Occured', incident_state)
    Incident_type = st.selectbox('Type of Incident', incident_type)
    vehicles_involved = st.slider('No. of Vehicles Involved', min_value=1, max_value=5, step=1)
    
    input = st.button("Predict Total Insurance Amount")
    
    if input:
        input = np.array([[Age, Authorities, Auto_Model, Collision_type, Incident_severity, Incident_state, Incident_type, vehicles_involved]])
        input[:, 1] = authorities_contacted_encode.transform( input[:, 1])
        input[:, 2] = auto_model_encode.transform( input[:, 2])
        input[:, 3] = collision_type_encode.transform( input[:, 3])
        input[:, 4] = incident_severity_encode.transform(input[:, 4])
        input[:, 5] = incident_state_encode.transform(input[:, 5])
        input[:, 6] = incident_type_encode.transform(input[:, 6])
        

        amount = model.predict(input)
        st.subheader(f"The estimated insurance amount is ₦{amount[0]:.2f}")
    