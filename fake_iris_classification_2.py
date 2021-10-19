import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

iris_data = datasets.load_iris()

class_names = ["setosa", "versicolor", "virginica"]

try:
    fake_data = pd.read_csv("fake_iris_2.csv")
except:

    d1 = pd.DataFrame(iris_data.data, columns=["sepal length", "sepal width", "petal length", "petal width"])
    # print(d1)
    d2 = pd.DataFrame(iris_data.target, columns=["species"])
    # print(d2)

    d3 = d1.join(d2)
    # print(d3)

    setosa = [(d3[:25]["sepal length"].mean(),d3[:25]["sepal length"].std()),
              (d3[:25]["sepal width"].mean(),d3[:25]["sepal width"].std()),
              ((d3[:25]["petal length"]/d3[:25]["sepal length"]).mean(),
               (d3[:25]["petal length"]/d3[:25]["sepal length"]).std()),
              ((d3[:25]["petal width"]/d3[:25]["sepal width"]).mean(),
               (d3[:25]["petal width"]/d3[:25]["sepal width"]).std())]
    versicolor = [(d3[50:75]["sepal length"].mean(),d3[50:75]["sepal length"].std()),
              (d3[50:75]["sepal width"].mean(),d3[50:75]["sepal width"].std()),
              ((d3[50:75]["petal length"]/d3[50:75]["sepal length"]).mean(),
               (d3[50:75]["petal length"]/d3[50:75]["sepal length"]).std()),
              ((d3[50:75]["petal width"]/d3[50:75]["sepal width"]).mean(),
               (d3[50:75]["petal width"]/d3[50:75]["sepal width"]).std())]
    virginica = [(d3[100:125]["sepal length"].mean(),d3[100:125]["sepal length"].std()),
              (d3[100:125]["sepal width"].mean(),d3[100:125]["sepal width"].std()),
              ((d3[100:125]["petal length"]/d3[100:125]["sepal length"]).mean(),
               (d3[100:125]["petal length"]/d3[100:125]["sepal length"]).std()),
              ((d3[100:125]["petal width"]/d3[100:125]["sepal width"]).mean(),
               (d3[100:125]["petal width"]/d3[100:125]["sepal width"]).std())]
    # (mean, stdev) pairs for normal distribution

    np.random.seed(1)
    sep_len_set = np.random.normal(setosa[0][0], setosa[0][1], 10 ** 6)
    sep_wid_set = np.random.normal(setosa[1][0], setosa[1][1], 10 ** 6)

    setosas = pd.DataFrame([sep_len_set,sep_wid_set,
                         sep_len_set*np.random.normal(setosa[2][0], setosa[2][1], 10 ** 6),
                         sep_wid_set*np.random.normal(setosa[3][0], setosa[3][1], 10 ** 6)],
                        index=["sepal length", "sepal width", "petal length", "petal width"])

    setosas = setosas.T  # transponse

    sep_len_vers = np.random.normal(versicolor[0][0], versicolor[0][1], 10 ** 6)
    sep_wid_vers = np.random.normal(versicolor[1][0], versicolor[1][1], 10 ** 6)

    versicolors = pd.DataFrame([sep_len_vers, sep_wid_vers,
                         sep_len_vers*np.random.normal(versicolor[2][0], versicolor[2][1], 10 ** 6),
                         sep_wid_vers*np.random.normal(versicolor[3][0], versicolor[3][1], 10 ** 6)],
                        index=["sepal length", "sepal width", "petal length", "petal width"])

    versicolors = versicolors.T

    sep_len_vir = np.random.normal(virginica[0][0], virginica[0][1], 10 ** 6)
    sep_wid_vir = np.random.normal(virginica[1][0], virginica[1][1], 10 ** 6)

    virginicas = pd.DataFrame([sep_len_vir, sep_wid_vir,
                         sep_len_vir*np.random.normal(virginica[2][0], virginica[2][1], 10 ** 6),
                         sep_wid_vir*np.random.normal(virginica[3][0], virginica[3][1], 10 ** 6)],
                        index=["sepal length", "sepal width", "petal length", "petal width"])

    virginicas = virginicas.T

    new_iris = [setosas, versicolors, virginicas]
    new_iris = pd.concat(new_iris, ignore_index=True)

    a1 = [0 for i in range(10 ** 6)]
    a2 = [1 for i in range(10 ** 6)]
    a3 = [2 for i in range(10 ** 6)]

    labels = [pd.DataFrame(a1,columns=["species"]), pd.DataFrame(a2,columns=["species"]), pd.DataFrame(a3,columns=["species"])]
    labels = pd.concat(labels, ignore_index=True)
    # print(labels)

    fake_data = new_iris.join(labels)
    fake_data = fake_data[(fake_data["sepal length"]>0) & (fake_data["sepal width"]>0) & (fake_data["petal length"]>0) & (fake_data["petal width"]>0)]
    # print(data2)
    file = fake_data.to_csv("fake_iris_2.csv",index=False)

#print(fake_data)

x_train = np.array(fake_data[["sepal length", "sepal width", "petal length", "petal width"]])
#print(x_train)

y_train = np.array(fake_data["species"])
#print(y_train)

x_test = np.array(iris_data.data)
y_test = np.array(iris_data.target)

#print(x_test)
#print(y_test)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


prediction = model.predict(x_test)
for i in range(len(x_test)):
    print("Prediction: ", class_names[prediction[i]], "Actual:", class_names[y_test[i]])


