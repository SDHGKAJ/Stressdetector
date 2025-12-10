import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
import joblib

files = glob.glob('data/processed/features/dataset_*.csv')
if not files:
    print('No labeled datasets found in data/processed/features/')
    raise SystemExit

dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=['iris_norm','openness_mean','blink_rate'])
X = data[['iris_norm','iris_std_px','openness_mean','blink_rate']].fillna(0).values
label_map = {'low':0,'med':1,'high':2}
y = data['label'].map(label_map).values
groups = (data['ts'].astype(int)%1000).values
clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
cv = GroupKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=cv.split(X,y,groups), scoring='f1_macro')
print('F1 macro:', scores.mean(), scores.std())
clf.fit(X,y)
joblib.dump(clf, 'models/cogload_model.joblib')
print('Model saved to models/cogload_model.joblib')
