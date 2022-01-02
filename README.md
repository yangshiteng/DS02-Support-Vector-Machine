# Introduction

Support vector machine is highly preferred by many as it produces significant accuracy with less computation power. Support Vector Machine, abbreviated as SVM can be used for both regression and classification tasks. But, it is widely used in classification objectives.

## What is Support Vector Machine?

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![image](https://user-images.githubusercontent.com/60442877/147865845-861a8b26-522a-4554-ad73-068df35cb2f1.png)

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

## Hyperplanes and Support Vectors

![image](https://user-images.githubusercontent.com/60442877/147866096-ef674ee5-c67f-4167-930e-c8a06d6fbe47.png)

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.

![image](https://user-images.githubusercontent.com/60442877/147866107-d8537d2a-7b13-4452-a0dd-bfa9cd1a3e0e.png)

Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

## Large Margin Intuition

In logistic regression, we take the output of the linear function and squash the value within the range of [0,1] using the sigmoid function. If the squashed value is greater than a threshold value(0.5) we assign it a label 1, else we assign it a label 0. In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.

## Cost Function and Gradient Updates

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












