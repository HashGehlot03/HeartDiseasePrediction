import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pickle
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title = 'Heart Analysis',page_icon='❤️')
# st.markdown(
#            """ <style> .main { background-color : #FE8E86; } </style> """,
#            unsafe_allow_html = True
# )
st.markdown("""
<style>
sidebar .sidebar-content {
    background-image: linear-gradient(#FE8E86,#FE8E86);
}
.main {
    background-color : #FE8E86;
}

</style>
    """, unsafe_allow_html=True)
hide_st_style = """
           <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
@st.cache
def load_data():
    return pd.read_csv('heart.csv')
df = load_data()

feature_description = '''
        |column|Description|
        |------|-----------|
        |age|Age of Patient|
        |sex|Gender|
        |cp|chest pain type (4 values)|
        |trestbps|resting blood pressure|
        |chol|serum cholestrol in mg/dl|
        |fbs|fasting blood sugar > 120 mg/dl|
        |restecg|resting electrocardiographic results (values 0,1,2)|
        |thalachh|maximum heart rate achieved|
        |exang|exercise induced angina|
        |oldpeak|ST depression induced by exercise relative to rest|
        |slope|the slope of the peak exercise ST segment|
        |ca|number of major vessels (0-3) colored by flourosopy|
        |thal|thal: 3 = normal; 6 = fixed defect; 7 = reversable defect|
        |target|having heart disease or not|
'''
model = pickle.load(open('HeartDiseasePredictor', 'rb'))
df_sex = df.sex.replace({0:'Male',1:'Female'})
df_cp = df.cp.replace({0:'Mild',1:'Gentle',2:'Tolerant',3:'Severe'})
#round(len(df.loc[df.output == 1,'output'])/len(df) * 100,2)
output_1 = df.loc[df.output == 1,:]
output_1.sex.replace({0:'Male',1:'Female'},inplace = True)
output_0 = df.loc[df.output == 0,:]
output_0.sex.replace({0:'Male',1:'Female'},inplace = True)
cont_feat = [' ','age','trtbps','chol','thalachh','oldpeak']
#output_1.loc[output_1.age >=55,'age']


#figures
fig_bar1 = px.histogram(output_1,x='age',nbins = 5,labels={'age':'Age','count':'No of persons'},color='sex',title = '<b>Survived</b>')
fig_bar0 = px.histogram(output_0,x='age',nbins = 5,labels={'age':'Age','count':'No of persons'},color='sex',title = '<b>Not Survived</b>')
fig_sex = px.histogram(df_sex,x='sex',color='sex',title = '<b>Male vs Female (%)</b>')
fig_pie = px.pie(df_cp,names = 'cp',title = '<b>Chest Pain (%)</b>')
fig_box = px.box(df,y='chol',color = 'output')     #Box plots for each continuous columns ['age','trtbps','chol','thalachh','oldpeak']




st.markdown('# Heart Disease Prediction')
st.image('heartimage.jpg')
side = st.sidebar.selectbox('Content',['Home','View Dashboard','Predict Disease','More Projects'])


if side == 'Home':
    head = st.container()
    mid = st.container()
    with head:
        
        st.title('Dataset')
        st.dataframe(df.sample(frac=1).iloc[0:20,:])
    with mid:
        col1,col2 = st.columns(2)
        with col1.expander('Details about the features'):
            st.markdown(feature_description)
        col2.download_button(label = 'Download Dataset to play with it',data=df.to_csv() ,file_name= 'Heart.csv',mime = 'text/csv')
        st.markdown('## Scope of the project')
        st.markdown("It's kind of Data Science Project which includes **Data Analysis Dashboard** ,**Predicting The Heart Disease** and **More Projects like this**. You can see the dataset and related code on [kaggle](https://www.kaggle.com/ronitf/heart-disease-uci).")


if side == 'View Dashboard':
    head = st.container()
    mid = st.container()
    with head:
        st.title('Visualize the Dataset')
        st.markdown(""" # Plot 1 :- """)
        st.markdown(""" ## Survived with their ages""")
        st.plotly_chart(fig_bar1)
        st.markdown(""" # Plot 2 :- """)
        st.markdown(""" ## Not Survived with their ages""")
        st.plotly_chart(fig_bar0)
        st.markdown(""" # Plot 3 :- """)
        st.markdown(""" ## Male vs Female Distributions""")
        st.plotly_chart(fig_sex)
        st.markdown(""" # Plot 4 :- """)
        st.markdown(""" ## Peoples having chest pain (levels)""")
        st.plotly_chart(fig_pie)
        st.markdown(""" # Plot 5 :- """)
        st.markdown(""" ## Box Plots""")
        cont_feature = st.selectbox('Select the feature',options = cont_feat)
        if cont_feature == ' ':
            pass
        else:
            fig_box = px.box(df,y=cont_feature,color = 'output')
            st.plotly_chart(fig_box)



if side == 'Predict Disease':
    st.markdown("# Predict the disease")
    age = int(st.number_input('Enter the Age'))
    sex = st.selectbox('Gender',options = ['Male','Female'])
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    cp = st.selectbox('Type of Chest Pain',options = ['Severe','Tolerant','Gentle','Mild'])
    if cp == 'Severe':
        cp = 3
    elif cp == 'Tolerant':
        cp = 2
    elif cp == 'Gentle':
        cp = 1
    elif cp == 'Mild':
        cp = 0

    trtbps = st.number_input(label = 'Resting Blood Pressure',min_value = 80,max_value=210)
    chol = st.number_input(label = 'Cholestrol Level',min_value=120,max_value=570)
    fbs = int(st.checkbox(label = 'Fasting blood pressure > 120 mg/dl ?'))
    restecg = st.selectbox(label = 'Resting electrocardiagraphic Results',options = [0,1,2])
    thalachh = st.number_input(label = 'Maximum Heart Achieved',min_value = 70,max_value=210)
    exng = int(st.checkbox(label = 'Exercise induced angina ?'))
    oldpeak = st.number_input(label = 'depression induced by exercise relative to rest',min_value = 0,max_value=7)
    slope = st.selectbox(label = 'the slope of the peak exercise ST segment',options = [0,1,2])
    caa = st.selectbox(label = 'number of major vessels',options = [0,1,2,3,4])
    thall = st.selectbox(label = 'thal',options = [0,1,2,3])

    if st.button('Predict'):
        result = model.predict([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slope,caa,thall]])
        if result[0] == 0:
            st.success('According to the predictions, you dont have a heart disease')
        else:
            st.error('According to the predictions, check you have symptoms of heart disease')


if side == 'More Projects':
    st.markdown("# My Projects ")
    st.markdown("### [Travel Package Purchase Prediction](https://share.streamlit.io/hashgehlot03/travelprediction/main/travelapp.py)")
    st.markdown("### [Water Potability Prediction](https://share.streamlit.io/hashgehlot03/waterpotabilityprediction/main/WaterPotabilityApp.py)")
    st.markdown("### [Book Recommendation System](https://share.streamlit.io/hashgehlot03/bookrecommendationsystem/main/Book_Recommendation_System.py)")
    st.markdown("### [Coronary Disease Prediction](https://coronary-disease-prediction.herokuapp.com)")
    st.markdown("### [Promotion Prediction](https://promotionprediction-app.herokuapp.com)")
    st.markdown("### [Trade Data Analysis](https://tradedashboard.herokuapp.com)")
    st.markdown('## You can also check my [Github](https://github.com/HashGehlot03) as well as [Linkedin](https://www.linkedin.com/in/harish-gehlot-5338a021a)')