from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd

_data = pd.read_csv('breast-cancer-wisconsin.data')

_data_new = _data.drop(['1000025'],axis=1)

X = _data_new.iloc[:,0:8]
Y = _data_new.iloc[:,9]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()

model.add(Dense(8, input_dim=8))
model.add(Dropout(0.1))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation("relu"))

model.compile(loss='mae', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=250, batch_size=16, validation_split=0.13)
test_loss, test_acc = model.evaluate(X_test, Y_test)

predictions = model.predict(X_test[:])
print('prediction:', predictions)

print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)
