import joblib

d = joblib.load('work_dir/model.joblib')
print(d)
print(d.classifier)