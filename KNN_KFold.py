# Group Project # 2
# K-fold cross-validation for hyper-parameter tuning and model comparison
#
# Implementation of K-Fold cross-validation
#   Used it with machine learning algorithms:
#       (1) train hyper-parameters
#       (2) compare prediction accuracy.

# imports
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as skKnn
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from gradient_descent import gradient_descent

num_neigh = 5

def knn_wrap( X_train, y_train, X_new ) :
    num_neighbors = num_neigh
    lab_enc = preprocessing.LabelEncoder()
    Y_encoded = lab_enc.fit_transform(y_train)
    knn = skKnn(n_neighbors=num_neighbors)
    knn.fit(X_train, Y_encoded)
    return knn.predict_proba( X_new )

# Function: KFoldCV
# input arguments:
#   X_mat, a matrix of numeric inputs (one row for each observation, one column for each feature).
#   y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
#   ComputePredictions, a function that takes three inputs (X_train,y_train,X_new),
#                       trains a model using X_train,y_train,
#                       then outputs a vector of predictions (one element for every row of X_new).
#   fold_vec, a vector of integer fold ID numbers (from 1 to K).
def KFoldCV( X_mat, y_vec, ComputePredictions, fold_vec ):

    X_mat = np.array(X_mat)
    y_vec = np.array(y_vec)

    K = np.max(fold_vec)
    #K = fold_vec.shape[0]
    X_size = X_mat.shape[0]

    # The function should begin by initializing a variable called error_vec, a numeric vector of size K.
    error_vec = np.empty( K )
    new_y_mat = []
    new_pred_mat = []

    #id_vec = np.random.randint( 1, K+1, X_mat.shape[ 0 ] )

    # The function should have a for loop over the unique values k in fold_vec (should be from 1 to K).
    for value in range( 1, K+1 ) :
    #for fold_id in fold_vec :
        X_new = []
        Y_new = []
        X_train = []
        y_train = []

        # first define X_new,y_new based on the observations for which the corresponding elements of fold_vec
        #   are equal to the current fold ID k.
        for index in range(X_mat.shape[0]):
            if (value == fold_vec[index]):
                X_new.append(X_mat[index].tolist())
                Y_new.append(y_vec[index].tolist())
                #X_new = np.append(X_new, X_mat[index])
                #Y_new = np.append(Y_new, y_vec[index])

            # then define X_train,y_train using all the other observations
            else:
                X_train.append(X_mat[index].tolist())
                y_train.append(y_vec[index].tolist())
                #X_train = np.append(X_train, X_mat[index])
                #y_train = np.append(y_train, y_vec[index])

        # then call ComputePredictions
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_new = np.array(X_new)
        pred_new = ComputePredictions( X_train, y_train, X_new )

        if(pred_new.ndim > 1):
            pred_new = pred_new[:,1]

        new_y_mat.append( Y_new )
        new_pred_mat.append( pred_new )

        pred_new = np.around(pred_new)

        # then compute the zero-one loss of pred_new with respect to y_new
        #   and store the mean (error rate) in the corresponding entry of error_vec.
        error_vec[value-1] = np.mean(Y_new != pred_new)

    # At the end of the algorithm you should return error_vec.
    #print("omg", error_vec)
    return(error_vec, new_y_mat, new_pred_mat )

X_mat = np.array([[1,2,3],
                  [3,2,1],
                  [1,2,3],
                  [1,2,3],
                  [3,2,1],
                  [1,2,3]
                  ])
Y_vec = np.array([1,0,1,1,0,1])
#print(KFoldCV( X_mat, Y_vec, knn_wrap, fold_vec))

# function: NearestNeighborsCV
# input arguments
# X_mat, a matrix of numeric inputs/features (one row for each observation, one column for each feature).
# y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
# X_new, a matrix of numeric inputs/features for which we want to compute predictions.
# num_folds, default value 5.
# max_neighbors, default value 20.
def NearestNeighborsCV( X_mat, y_vec, X_new, num_folds=5, max_neighbors=20 ):
    global num_neigh
    # randomly create a variable called validation_fold_vec, a vector with integer values from 1 to num_folds.
    #validation_fold_vec = np.arange( 1, num_folds+1 )
    validation_fold_vec = np.random.randint(1, num_folds + 1, X.shape[0])
    #print(validation_fold_vec)

    # initialize a variable called error_mat, a numeric matrix (num_folds x max_neighbors).
    error_mat = np.empty([num_folds, max_neighbors])

    # There should be a for loop over num_neighbors from 1 to max_neighbors.
    for index in range( 1, max_neighbors+1 ):

        # Inside the for loop you should call KFoldCV, and specify ComputePreditions=a function that uses your
        #   programming language’s default implementation of the nearest neighbors algorithm, with num_neighbors.
        #   e.g. scikit-learn neighbors in Python, class::knn in R.
        #   Store the resulting error rate vector in the corresponding column of error_mat.
        num_neigh = index
        error_col = KFoldCV( X_mat, y_vec, knn_wrap, validation_fold_vec )
        error_mat[:, index-1] = error_col[0][:]

    # Compute a variable called mean_error_vec (size max_neighbors) by taking the mean of each column of error_mat.
    mean_error_vec = np.empty(max_neighbors)
    for index in range(mean_error_vec.shape[0]):
        mean_error_vec[index] = np.mean( error_mat[:,index] )

    # Compute a variable called best_neighbors which is the number of neighbors with minimal error.
    best_neighbors = np.argmin(mean_error_vec)

    num_neigh = best_neighbors+1
    pred_new = knn_wrap(X_mat, y_vec, X_new )

    # Your function should output
    #   (1) the predictions for X_new, using the entire X_mat,y_vec with best_neighbors;
    #   (2) the mean_error_mat for visualizing the validation error.
    return(pred_new, mean_error_vec, error_mat, best_neighbors)

def NearestNeighborsCV_Wrapper( X_mat, y_vec, X_new ):
    results = NearestNeighborsCV( X_mat, y_vec, X_new )
    return results[0]

def BaselineFunciton( X_mat, y_vec, X_new ) :
    X_new = np.array(X_new)
    return np.zeros(X_new.shape[0])

def Gradient_Wrapper( X_mat, y_vec, X_new):
    weightVec = gradient_descent( X_mat, y_vec, 0.1, 600 )
    return np.dot(X_new, weightVec)

# First scale the inputs (each column should have mean 0 and variance 1).
#   You can do this by subtracting away the mean and then dividing by the standard deviation of each column
#   (or just use a standard function like scale in R).
# read data from csv
all_data = np.genfromtxt('spam.data', delimiter=" ")
# get size of data
size = all_data.shape[1] - 1
# set inputs to everything but last col, and scale
X = scale(np.delete(all_data, size, axis=1))
# set outputs to last col of data
y = all_data[:, size]

# Use NearestNeighborsCV on the whole data set, then plot validation error as a function of the number of neighbors,
#   separately for each fold.
# Draw a bold line for the mean validation error, and draw a point to emphasize the minimum.
X_new = np.zeros((X.shape[0],X.shape[1]))
num_folds = 5
num_neighbors = 20
pred, vec, mat, best_neigh = NearestNeighborsCV(X, y, X_new, num_folds, num_neighbors)
x_axis = np.arange(0, num_neighbors)
x_axis_labels = x_axis + 1


plt.figure()
for fold in range(num_folds):
    plt.plot( mat[fold], "b-", linewidth=0.5)
plt.plot( vec, "g-", label="mean validation error")
plt.plot( best_neigh, vec[best_neigh], "ro" )
plt.xticks(x_axis, x_axis_labels)
plt.ylim(0)
plt.legend()
plt.title("Validation Error")
plt.ylabel("Percent Error")
plt.xlabel("Number of Neighbors")
plt.show()


# Randomly create a variable test_fold_vec which is a vector with one element for each observation,
#   and elements are integers from 1 to 4. In your report please include a table of counts with a row for each fold (1/2/3/4) and a column for each class (0/1)
test_fold_vec = np.random.randint(1, 4 + 1, X.shape[0])

def count_fold( fold ) :
    count0, count1 = 0, 0
    for index, value in enumerate( test_fold_vec ) :
        if( fold == value ) :
            if(y[index] == 0): count0 += 1
            else: count1 += 1
    return(count0,count1)

print("{: >11} {: >4} {: >4}".format("folds", "0", "1"))
count = count_fold(1)
print("{: >11} {: >4} {: >4}".format("1", count[0], count[1]))
count = count_fold(2)
print("{: >11} {: >4} {: >4}".format("2", count[0], count[1]))
count = count_fold(3)
print("{: >11} {: >4} {: >4}".format("3", count[0], count[1]))
count = count_fold(4)
print("{: >11} {: >4} {: >4}".format("4", count[0], count[1]))

# Use KFoldCV with three algorithms:
#   (1) baseline/underfit – predict most frequent class
result_base = KFoldCV(X, y, BaselineFunciton, test_fold_vec)
base_labels = ['Base', 'Base', 'Base', 'Base']
plt.scatter( result_base[0], base_labels )

#   (2) NearestNeighborsCV,
result_nearest = KFoldCV(X, y, NearestNeighborsCV_Wrapper, test_fold_vec, "b-")
nearest_labels = ['Nearest', 'Nearest', 'Nearest', 'Nearest']
plt.scatter( result_nearest[0], nearest_labels )

#   (3) overfit 1-nearest neighbors model. Plot the resulting test error values as a function of the data set,
#       in order to show that the NearestNeighborsCV is more accurate than the other two models.
num_neigh = 1
result_over = KFoldCV(X, y, knn_wrap, test_fold_vec, "g-")
over_labels = ['Overfit', 'Overfit', 'Overfit', 'Overfit']
plt.scatter( result_over[0], over_labels )

#   (4) Gradient Descent
result_grad = KFoldCV(X, y, Gradient_Wrapper, test_fold_vec, "g-")
grad_labels = ['Gradient', 'Gradient', 'Gradient', 'Gradient']
plt.scatter( result_grad[0], grad_labels )

plt.title("Percent of Error Comparison")
plt.xlim(left=0)
plt.show()

base_label = "Baseline"
knn_label = "K Nearest Neighbor"
over_label = "Overfit KNN"
grad_label = "Gradient Descent"

for index in range(4) :
    fpr, tpr, thr = roc_curve(result_base[1][index], result_base[2][index])
    plt.plot(fpr,tpr,"r-",linewidth=".4",label=base_label)

    fpr, tpr, thr = roc_curve(result_nearest[1][index], result_nearest[2][index])
    plt.plot(fpr,tpr,"b-",linewidth=".6",label=knn_label)
    plt.plot(fpr[int(len(fpr)/2)], tpr[int(len(tpr)/2)], "b.")

    fpr, tpr, thr = roc_curve(result_over[1][index], result_over[2][index])
    plt.plot(fpr, tpr, "g-",linewidth=".4",label=over_label)
    plt.plot(fpr[int(len(fpr)/2)], tpr[int(len(tpr)/2)], "g.")

    fpr, tpr, thr = roc_curve(result_grad[1][index], result_grad[2][index])
    plt.plot(fpr, tpr, "c-", linewidth=".4", label=grad_label)
    plt.plot(fpr[int(len(fpr) / 2)], tpr[int(len(tpr) / 2)], "c.")

    base_label = "_nolegend_"
    knn_label = "_nolegend_"
    over_label = "_nolegend_"
    grad_label = "_nolegend_"

plt.title("ROC curves, 4-fold CV")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.legend()
plt.show()

for index in range(4) :
    score_base = roc_auc_score(result_base[1][index], result_base[2][index])
    plt.scatter( score_base, "Baseline", c="r" )

    score_nearest = roc_auc_score(result_nearest[1][index], result_nearest[2][index])
    plt.scatter( score_nearest, "Nearest", c="b" )

    score_over = roc_auc_score(result_over[1][index], result_over[2][index])
    plt.scatter( score_over, "Overfitting", c="g")

    score_grad = roc_auc_score(result_grad[1][index], result_grad[2][index])
    plt.scatter( score_grad, "Gradient", c="c")

plt.title("ROC Area Under Curve Comparison")
plt.xlim(right=1)
plt.show()