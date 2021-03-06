# Genome-sequence


New dataset source: http://www.cs.toronto.edu/~delve/data/splice/desc.html -> https://openml.org/d/46 for the
csv format
Using R programming, I perform exploratory analysis:
1. Probability distribution
2. Dataset Domain
3. Maximum and Minimum
4. Central tendencies
5. Measure of dispersion
6. Constant
7. Missing values and outliers
8. Data correlation between features


individual classifiers:
1. Naïve Bayes
2. Support Vector Machine
3. Decision Tree
4. Nearest Neighbor


I work with three ensemble methods to my dataset above:
1. Random Forest
2. Boosting
3. Stacking

This is an example of my analysis to the model: 
![image](https://user-images.githubusercontent.com/71423299/157905025-f3f8272a-98e9-4e62-b518-4815453e3ed4.png)
![image](https://user-images.githubusercontent.com/71423299/157905096-9a660228-3134-454f-b611-af2318402c59.png)



For this project – I used the parallel processing to enhance the process speed.
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

Unfortunately, using one-hot-encoded data made the data leakage during cross-validation, but I
performed cross-validation folding in Generalized-boosting model and this could be why it performed
better than XGBoost as the cross validation is very effective on reducing bias as the model use the most
of the data for fitting, while reduces variance by using the most data for validation set. So it balances out
bias and variances in an useful way.


![image](https://user-images.githubusercontent.com/71423299/158017619-bd994aec-8dd6-486c-91c5-07b26531ae77.png)

![image](https://user-images.githubusercontent.com/71423299/158017616-a324f1ff-7b6c-4a94-9e06-3960f94a40e7.png)

