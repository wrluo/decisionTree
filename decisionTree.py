import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import metrics

from sklearn.externals.six import StringIO  
import pydot

from IPython.display import Image

feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

df = pd.io.parsers.read_csv(
    filepath_or_buffer='iris.data',
    header=None,
    sep=',',
    )
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

X = df[[l for i,l in sorted(feature_dict.items())]].values
y = df['class label'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# y_predict = clf.predict(X)
# print(X)

# accuragy = metrics.accuracy_score(y, y_predict)
# print(accuragy)

# y_predict = clf.predict_proba(X)
# print(y_predict)