# AI Project 2023 ShopEasy
**Team Members:** Marek Kmoch (286021) &amp; Florian Neuhaus (286001)  
### [Section 1] Introduction  
In this project, we've been tasked with analyzing a dataset provided by ShopEasy, an e-commerce company. 
This dataset contains various types of information, including the total costs of items purchased by users and their monthly payment amounts.
Our role involved processing this data to distinguish between essential and trivial details, and then categorizing ShopEasy's customers 
accordingly. The goal is to enable ShopEasy to enhance user experiences by tailoring special promotions and improving services. 
Leveraging machine learning and our customer categorization, ShopEasy can now more effectively target specific customer segments for 
their promotional efforts, based on the combination of all of the detailed user information. 
### [Section 2] Methods
We've started by familiarizing ourselves with all of the data columns and what they mean. Right away it was apparent that frequencyIndex,
itemBuyFrequency and webUsage are all very similar and that down the line we would have to something about these. When we started with 
exploratory data analysis, it was important to make everything as clear and readable as possible. There was a lot of variety between some 
user data, so it was important to display all lables clearly, especially during plot distribution creation. The main focus point during this
part was NULL value rows, as there were 314 such rows, which was 3.5% of the whole dataset. Following detailed analysis, all of the NULL 
values were dropped because they generally followed the distribution of the data and wouldn't make a significant different during training.  

Preprocessing the dataset was an easy task. The personId column is a unique identifier, so every row is "unique" in its own sense, but we 
still ended up checking for duplicate personId values just to be safe (there weren't any). We also decided to encode categorical features at 
the beginning of EDA since they also offer an insight into analyzing NULL values and the general distribution of the data, so preprocessing
was pretty much already finished.  

The project is definitely a clustering problem, as the task asks "by applying segmentation to this dataset, ShopEasy aims to uncover these 
hidden patterns". Based on this statement, we know that a method that involves organizing something into categories will have to used, so 
regression is taken out of the equation because it predicts a continous value and not a group. The two final ones are clustering VS. Classification.
Going back to the statement, there is emphasis on "hidden patterns", which means that the groups are not known. Classification focuses on outcomes
that it knows, such as if something is a cat - yes or no. This means we use clustering because we do not know the labels, the groups or even the 
amount of groups that the customers could be divided into based on their personal data.  

The method choice came down to these three: KMeans, DBScan and Hierarchical Clustering. KMeans and Hierarchical clustering are both linear 
methods, which means that the methods assume that clusters are formed in circular shapes and don't have non-traditional shapes, therefore one
of these will be chosen. DBScan is a non-linear method since it can identify varying cluster shapes much more efficiently, therefore this will
100% be one of the two methods, as we do not know at all how the clusters will look and what their shapes and sizes will be like. Between 
KMeans and Hierarchical clustering, we opted to go with KMeans because it is considered the more efficient of the two methods. It is also 
much more scalable, which is important for our big dataset of 8950 (excluding the deleted NULL value rows). Therefore, we decided to go with
KMeans (more specifically KMeans++ and DBScan). 

### [Section 3] Experimental Design  
### [Section 4] Results  
### [Section 5] Conclusions  
