import streamlit as st 

import pandas as pd
import numpy as np
from PIL import Image

st.title("Gender Classifier")
html_temp = """
	<div style="background-color:DeepSkyBlue;padding:10px;border: 5px solid  SkyBlue;border-radius: 15px 50px">
	<h2 style="color:white;text-align:center;">Predict your gender..</h2>
	</div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)
load_css('icon.css')
load_icon('people')

def load_images(file_name):
	img = Image.open(file_name)
	return st.image(img,width=250)

st.header('EDA')
st.subheader('Name DataSet')

myData = 'names_dataset.csv'
@st.cache(allow_output_mutation=True)
def load_data(myData):
    data = pd.read_csv(myData)
    return data


data = load_data(myData)

# df_names = data
# df_names.sex.replace({'F':0,'M':1},inplace=True)

if st.checkbox('Show Dataset'):
    if st.button('Head'):
        st.write(data.head())
    if st.button('Tail'):
        st.write(data.tail())

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

features = np.vectorize(features)
df_X = features(data['name'])
df_y = data['sex']

dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dv = DictVectorizer()
vectorizer = dv.fit_transform(dfX_train)

from sklearn.tree import DecisionTreeClassifier
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)



def predict_gender(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    result = dclf.predict(vector)
    return result

name = st.text_input("Enter Name")
if st.button("Predict"):
    result = predict_gender(name)
    if result[0] == 0:
        prediction = 'Female'
        c_img = 'female.png'
    else:
        prediction = 'Male'
        c_img = 'male.png'
	
    st.success('Name: {} was classified as {}'.format(name.title(),prediction))
    load_images(c_img)




