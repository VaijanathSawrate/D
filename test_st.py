import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from pickle import load
from pickle import dump

st.title('Model Deployment:Logistic Regression')

st.sidebar.header('User Input Parameters')



def user_input_features():
    CLMSEX=st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR=st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT=st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE=st.sidebar.number_input("Insert the Age")
    LOSS=st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features=pd.DataFrame(data,index=[0])
    return features


df=user_input_features()
st.subheader('User Input Parameters')
st.write(df)

claimants=pd.read_csv("D:\\DATA_science\\Data_sets\\Logistic_Reg\\claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis=1)
claimants=claimants.dropna()

X=claimants.iloc[:,[1,2,3,4,5]]
Y=claimants.iloc[:,0]
clf=LogisticRegression()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('No' if prediction_proba[0][1] > 0.5 else 'Yes')

st.subheader('Prediction Probability')
st.write(prediction_proba)





