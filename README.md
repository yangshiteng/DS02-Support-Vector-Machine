# 1. Introduction

Support vector machine is a supervised mahince learning algorithm and highly preferred by many as it produces significant accuracy with less computation power. Support Vector Machine, abbreviated as SVM can be used for  regression, classification tasks and outlier detection. But, it is widely used in classification objectives.

The advantages of support vector machines are:

- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation 

## 1.1 What is Support Vector Machine?

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![image](https://user-images.githubusercontent.com/60442877/147865845-861a8b26-522a-4554-ad73-068df35cb2f1.png)

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

## 1.2 Hyperplanes and Support Vectors

![image](https://user-images.githubusercontent.com/60442877/147866096-ef674ee5-c67f-4167-930e-c8a06d6fbe47.png)

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.

![image](https://user-images.githubusercontent.com/60442877/147866107-d8537d2a-7b13-4452-a0dd-bfa9cd1a3e0e.png)

Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

## 1.3 Large Margin Intuition

In logistic regression, we take the output of the linear function and squash the value within the range of [0,1] using the sigmoid function. If the squashed value is greater than a threshold value(0.5) we assign it a label 1, else we assign it a label 0. In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.

## 1.4 Cost Function and Gradient Updates

In the SVM algorithm, we are looking to maximize the margin between the data points and the hyperplane. The loss function that helps maximize the margin is hinge loss.

![image](https://user-images.githubusercontent.com/60442877/147866170-46de256b-922b-4308-8600-82b893eaa10f.png)

The above Hinge loss function can also be represented as, 

![image](https://user-images.githubusercontent.com/60442877/147866212-d5dac086-c7f3-46a6-8789-5d226e079e60.png)

The cost is 0 if the predicted value and the actual value are of the same sign. If they are not, we then calculate the loss value. We also add a regularization parameter to the cost function. The objective of the regularization parameter is to balance the margin maximization and loss. After adding the regularization parameter, the cost functions looks as below.

![image](https://user-images.githubusercontent.com/60442877/147866183-aa81a5d5-9ba5-4e78-b574-56ee8d16a658.png)

Now that we have the loss function, we take partial derivatives with respect to the weights to find the gradients. Using the gradients, we can update our weights.

![image](https://user-images.githubusercontent.com/60442877/147866219-be264017-c76e-48ff-a3fb-ef58f3bfa3b7.png)

When there is no misclassification, i.e our model correctly predicts the class of our data point, we only have to update the gradient from the regularization parameter.

![image](https://user-images.githubusercontent.com/60442877/147866227-6946ad84-cb28-4115-a3b0-e27cc177b67a.png)

When there is a misclassification, i.e our model make a mistake on the prediction of the class of our data point, we include the loss along with the regularization parameter to perform gradient update.

![image](https://user-images.githubusercontent.com/60442877/147866238-aa882eac-6d94-4682-83ef-069800bfb2ca.png)


## 1.5 Soft Margin

Two types of misclassifications are tolerated by SVM under soft margin:
- The dot is on the wrong side of the decision boundary but on the correct side/ on the margin (shown in left)
- The dot is on the wrong side of the decision boundary and on the wrong side of the margin (shown in right)

![image](https://user-images.githubusercontent.com/60442877/147867156-69b0158d-cb37-49e7-87c3-b05d3cb2c2ca.png)

Applying Soft Margin, SVM tolerates a few dots to get misclassified and tries to balance the trade-off between finding a line that maximizes the margin and minimizes the misclassification.

### 1.51 Degree of tolerance

How much tolerance(soft) we want to give when finding the decision boundary is an important hyper-parameter for the SVM (both linear and nonlinear solutions). In Sklearn, it is represented as the penalty term — ‘C’. The bigger the C, the more penalty SVM gets when it makes misclassification. Therefore, the narrower the margin is and fewer support vectors the decision boundary will depend on.

![image](https://user-images.githubusercontent.com/60442877/147867177-5eac8d16-8742-4306-ba2f-591f4bd5f683.png)

## 1.6 Kernel Trick 

What Kernel Trick does is that it utilizes existing features, applies some transformations, and creates new features. Those new features are the key for SVM to find the nonlinear decision boundary.

In Sklearn — svm.SVC(), we can choose ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable as our kernel/transformation. I will give examples of the two most popular kernels — Polynomial and Radial Basis Function(RBF).

![image](https://user-images.githubusercontent.com/60442877/147867614-33c201fa-8c1d-41d1-bf7f-38712e50efad.png)

- Polynomial Kernel

Think of the polynomial kernel as a transformer/processor to generate new features by applying the polynomial combination of all the existing features.

To illustrate the benefit of applying a polynomial transformer, let’s use a simple example:

![image](https://user-images.githubusercontent.com/60442877/147867666-04efbdec-e10c-4c5d-85e0-c44d3a943cb1.png)

Existing Feature: X = np.array([-2,-1,0, 1,2])
Label: Y = np.array([1,1,0,1,1])
it’s impossible for us to find a line to separate the yellow (1)and purple (0) dots (shown on the left).

But, if we apply transformation X² to get:
New Feature: X = np.array([4,1,0, 1,4])
By combing the existing and new feature, we can certainly draw a line to separate the yellow purple dots (shown on the right).

Support vector machine with a polynomial kernel can generate a non-linear decision boundary using those polynomial features.

- Radial Basis Function (RBF) kernel

Think of the Radial Basis Function kernel as a transformer/processor to generate new features by measuring the distance between all other dots to a specific dot/dots — centers. The most popular/basic RBF kernel is the Gaussian Radial Basis Function:

![image](https://user-images.githubusercontent.com/60442877/147867788-5e26b18a-75a3-4372-a590-cbe960b2cc4e.png)

gamma (γ) controls the influence of new features — Φ(x, center) on the decision boundary. The higher the gamma, the more influence of the features will have on the decision boundary, more wiggling the boundary will be.

To illustrate the benefit of applying a Gaussian rbf (gamma = 0.1), let’s use the same example:

![image](https://user-images.githubusercontent.com/60442877/147867795-8b1ca640-3836-430a-be66-926363abe57d.png)

Existing Feature: X = np.array([-2,-1,0, 1,2])
Label: Y = np.array([1,1,0,1,1])
Again, it’s impossible for us to find a line to separate the dots (on left hand).

But, if we apply Gaussian RBF transformation using two centers (-1,0) and (2,0) to get new features, we will then be able to draw a line to separate the yellow purple dots (on the right):
New Feature 1: X_new1 = array([1.01, 1.00, 1.01, 1.04, 1.09])
New Feature 2: X_new2 = array([1.09, 1.04, 1.01, 1.00, 1.01])

Similar to the penalty term — C in the soft margin, Gamma is a hyperparameter that we can tune for when we use SVM with kernel.

![image](https://user-images.githubusercontent.com/60442877/147867856-36b7c17f-8437-4cec-96b6-b0e13190cb78.png)

![image](https://user-images.githubusercontent.com/60442877/147867862-e67b60ae-1ae8-4c96-aeb3-2d5d54697c50.png)

To sum up, SVM in the linear nonseparable cases:

- By combining the soft margin (tolerance of misclassifications) and kernel trick together, Support Vector Machine is able to structure the decision boundary for linear non-separable cases.
- Hyper-parameters like C or Gamma control how wiggling the SVM decision boundary could be.
- the higher the C, the more penalty SVM was given when it misclassified, and therefore the less wiggling the decision boundary will be
- the higher the gamma, the more influence the feature data points will have on the decision boundary, thereby the more wiggling the boundary will be








# 2. Support Vector Machine for Regression

Support Vector Machines (SVMs) are well known in classification problems. The use of SVMs in regression is not as well documented, however. These types of models are known as Support Vector Regression (SVR). SVR is a powerful algorithm that allows us to choose how tolerant we are of errors, both through an acceptable error margin(ϵ) and through tuning our tolerance of falling outside that acceptable error rate. 

## 2.1 Simple Linear Regression

In most linear regression models, the objective is to minimize the sum of squared errors. Take Ordinary Least Squares (OLS) for example. The objective function for OLS with one predictor (feature) is as follows:

![image](https://user-images.githubusercontent.com/60442877/147866693-8be559b2-2a91-4880-9755-462519980d91.png)

where yᵢ is the target, wᵢ is the coefficient, and xᵢ is the predictor (feature).

![image](https://user-images.githubusercontent.com/60442877/147866699-3110e0a9-0702-4e62-bea5-8e81ff9b9127.png)

Lasso, Ridge, and ElasticNet are all extensions of this simple equation, with an additional penalty parameter that aims to minimize complexity and/or reduce the number of features used in the final model. Regardless, the aim — as with many models — is to reduce the error of the test set.

However, what if we are only concerned about reducing error to a certain degree? What if we don’t care how large our errors are, as long as they fall within an acceptable range?

Take housing prices for example. What if we are okay with the prediction being within a certain dollar amount — say $5,000? We can then give our model some flexibility in finding the predicted values, as long as the error is within that range.

## 2.2 Support Vector Regression

Enter Support Vector Regression. SVR gives us the flexibility to define how much error is acceptable in our model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.

In contrast to OLS, the objective function of SVR is to minimize the coefficients — more specifically, the l2-norm of the coefficient vector — not the squared error. The error term is instead handled in the constraints, where we set the absolute error less than or equal to a specified margin, called the maximum error, ϵ (epsilon). We can tune epsilon to gain the desired accuracy of our model. Our new objective function and constraints are as follows:

![image](https://user-images.githubusercontent.com/60442877/147866760-90f380fc-c85d-4fcc-add0-c6f9f3024413.png)

Let’s try the simple SVR on our dataset. The plot below shows the results of a trained SVR model on the Boston Housing Prices data. The red line represents the line of best fit and the black lines represent the margin of error, ϵ, which we set to 5 ($5,000).

![image](https://user-images.githubusercontent.com/60442877/147866774-e88d0242-968a-4c11-b725-94b575993f15.png)

You may quickly realize that this algorithm doesn’t work for all data points. The algorithm solved the objective function as best as possible but some of the points still fall outside the margins. As such, we need to account for the possibility of errors that are larger than ϵ. We can do this with slack variables.

The concept of slack variables is simple: for any value that falls outside of ϵ, we can denote its deviation from the margin as ξ. We know that these deviations have the potential to exist, but we would still like to minimize them as much as possible. Thus, we can add these deviations to the objective function.

![image](https://user-images.githubusercontent.com/60442877/147866816-6e9574f3-d1d0-4816-ace7-5fcb01ef4d81.png)

We now have an additional hyperparameter, C, that we can tune. 

Let’s set C=1.0 and retrain our model above. The results are plotted below

![image](https://user-images.githubusercontent.com/60442877/147866850-d576bfe8-0803-45f8-80e1-9873c90baf0f.png)

- Finding the Best Value of C

The above model seems to fit the data much better. We can go one step further and grid search over C to obtain an even better solution. Let’s define a scoring metric, % within Epsilon. This metric measures how many of the total points within our test set fall within our margin of error. We can also monitor how the Mean Absolute Error (MAE) varies with C as well.

Below is a plot of the grid search results, with values of C on the x-axis and % within Epsilon and MAE on the left and right y-axes, respectively.

![image](https://user-images.githubusercontent.com/60442877/147866887-8e43a075-f600-4a2f-86b4-bf46b8b9e5b8.png)

As we can see, MAE generally decreases as C increases. However, we see a maximum occur in the % within Epsilon metric. Since our original objective of this model was to maximize the prediction within our margin of error ($5,000), we want to find the value of C that maximizes % within Epsilon. Thus, C=6.13.

Let’s build one last model with our final hyperparameters, ϵ=5, C=6.13.

![image](https://user-images.githubusercontent.com/60442877/147866899-2446fbfb-ed51-4d29-9c88-49bc15e71d06.png)

The plot above shows that this model has again improved upon previous ones, as expected.



# 3. Video Tutorial 










