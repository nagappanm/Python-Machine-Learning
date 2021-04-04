import joblib
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
multilabel = MultiLabelBinarizer()

model = joblib.load('lps_model.pkl')
tfidf = joblib.load('tfidf_model.pkl')

df = pd.read_csv('https://raw.githubusercontent.com/nagappanm/Python-Machine-Learning/master/Multi_Label_Text_Classification_with_Skmultilearn/data/so_dataset_updated_blank.csv')
df['Tagsupdated']=df['Tagsupdated'].fillna("")
df['Tagsupdated'] = df['Tagsupdated'].apply(lambda x: x.split(','))

y = multilabel.fit_transform(df['Tagsupdated'])

x = [ 'how to write code in query and php Histogram:']

y_predict = model.predict(tfidf.transform(x))

print("The Output from model is:",(y_predict.toarray()));

print("The Output from model is:",multilabel.inverse_transform(y_predict));

inverseTransformList = multilabel.inverse_transform(y_predict)

out = [item.strip() for t in inverseTransformList for item in t]
out
from collections import OrderedDict
out = list(OrderedDict.fromkeys(out))
print("The List distinct is:",out)

list1 = [" test1","","test2 "]
newList = [item.strip() for item in list1]

print(list(filter(None, newList)))