# fake_iris_classification

This program creates a new data set from the existing and well known iris data set.

New data is made by finding mean and standandard deviation of all attributes (sepal length, sepal width, petal length and petal width) over all species (setosa, versicolor and virginica) and generating 10^6 new instances for every species using a normal distribution with those parameters. Any rows that have a negative value generated are not saved.

This data is then used to train the k nearest neighbors algorithm with 9 neighbours.

Regular iris data set is used for testing this model.

Finally, the program gives us the accuracy and the comparison between test data and prediction outputs.

Warning: This program generates a large (over 200MB) .csv file that stores the data, and that due to its size could'nt be uploaded here.

Update:

The new version creates the fake iris data set from one half of the original data, taking into consideration sepal length / petal length and sepal width / petal width ratio.

The model is the same and it is tested on the whole original iris data set. 
