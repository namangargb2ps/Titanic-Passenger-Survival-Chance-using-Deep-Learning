import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in Titanic Journey")
pclass=st.slider('Enter the Passender class fot the user',1,3)
sex=st.selectbox('Enter the Passenger Gender',['male','female'])
sibsp=st.slider('Enter the Passengers Sibling and spouse', 1,8)
parch=st.slider('Enter the Passengers total number of parents and child', 1,6)
fare=st.number_input('Enter the Fare of the Passenger')
embarked=st.selectbox('Enter the place Passenger started the Journey',['Southhampton', 'Queenstown' ,'Chebourg'])
data=pd.DataFrame([{'Pclass':pclass, 'Sex':sex, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare,
               'Embarked':embarked}])
model=load_model('model.h5')
with open('label_encoder.pkl', 'rb') as file:
    lb=pickle.load(file)
with open('one_hot_encoder.pkl', 'rb') as file:
    ohe=pickle.load(file)
with open ('scaler.pkl', 'rb') as file:
    std=pickle.load(file)

data['Sex']=lb.transform(data['Sex'])
embarked= ohe.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked, columns=ohe.get_feature_names_out())
data=pd.concat([data.drop(columns=['Embarked']), embarked], axis=1)
data[['Pclass','SibSp', 'Parch', 'Fare']]= std.transform(data[['Pclass','SibSp', 'Parch', 'Fare']])

y=model.predict(data)
y=y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger will survive the jounrey'
    else:
        return 'The Passenger wont survive the journey'
    
if st.button('Predict Survival chance'):
    st.write("Probability for Passenger's survival",y)
    st.write(Chance(y))
