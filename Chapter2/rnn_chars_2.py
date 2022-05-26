from keras.models import Sequential
from keras.layers import SimpleRNN, TimeDistributed, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

np.random.seed(1234)


data = ['xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz','pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr']


enc = LabelEncoder()
alphabet = np.array(list(set([c for w in data for c in w])))

enc.fit(alphabet)

int_enc=enc.fit_transform(alphabet)

onehot_encoder = OneHotEncoder(sparse=False)
int_enc=int_enc.reshape(len(int_enc), 1)
onehot_encoded = onehot_encoder.fit_transform(int_enc)


X_train=[]
y_train=[]

for w in data:
    for i in range(len(w)-1):
        X_train.extend(onehot_encoder.transform([enc.transform([w[i]])]))
        y_train.extend(onehot_encoder.transform([enc.transform([w[i+1]])]))

        
X_test=[]
y_test=[]

test_data = ['xyzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxyz','pqraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapqr']


for w in test_data:
    for i in range(len(w)-1):
        X_test.extend(onehot_encoder.transform([enc.transform([w[i]])]))
        print i,w[i],onehot_encoder.transform([enc.transform([w[i]])])
        y_test.extend(onehot_encoder.transform([enc.transform([w[i+1]])]))

        
sample_size=512
sample_len=len(X_train)

X_train = np.array([X_train*sample_size]).reshape(sample_size,sample_len,len(alphabet))
y_train = np.array([y_train*sample_size]).reshape(sample_size,sample_len,len(alphabet))


test_len=len(X_test)
X_test= np.array([X_test]).reshape(1,test_len,len(alphabet))
y_test= np.array([y_test]).reshape(1,test_len,len(alphabet))

model=Sequential()
model.add(SimpleRNN(input_dim  = len(alphabet), output_dim = 100, return_sequences = True))
model.add(TimeDistributed(Dense(output_dim = len(alphabet), activation  =  "sigmoid")))
model.compile(loss="binary_crossentropy",metrics=["accuracy"], optimizer = "adam")
model.fit(X_train, y_train, nb_epoch = 100, batch_size = 32)

preds=model.predict(X_test)[0] 

res=[]
for p in preds:
    m=np.argmax(p)
    res.extend(enc.inverse_transform(m))
print ''.join(res[0:len(test_data[0])])
print ''.join(res[len(test_data[0]):])



print(model.evaluate(X_test,y_test,batch_size=32)) 

