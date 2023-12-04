# AI Project 2023 ShopEasy
**Team Members:** Marek Kmoch (286021) &amp; Florian Neuhaus (286001)  
## [Section 1] Introduction  
In this project, we've been tasked with analyzing a dataset provided by ShopEasy, an e-commerce company. 
This dataset contains various types of information, including the total costs of items purchased by users and their monthly payment amounts.
Our role involved processing this data to distinguish between essential and trivial details, and then categorizing ShopEasy's customers 
accordingly. The goal is to enable ShopEasy to enhance user experiences by tailoring special promotions and improving services. 
Leveraging machine learning and our customer categorization, ShopEasy can now more effectively target specific customer segments for 
their promotional efforts, based on the combination of all of the detailed user information. 
## [Section 2] Methods
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
methods, which means that the methods assume that clusters are formed in circular shapes and don't have non-traditional shapes.
. DBScan is a non-linear method since it can identify varying cluster shapes much more efficiently, therefore, after our initial thought process, we thought
that we should choose of linear and one non-linear , as we do not know at all how the clusters will look and what their shapes and sizes will be like. Between 
KMeans and Hierarchical clustering, we opted to go with KMeans because it is considered the more efficient of the two methods. It is also 
much more scalable, which is important for our big dataset of 8950 (excluding the deleted NULL value rows). Therefore, we decided to go with
KMeans (more specifically KMeans++) and DBScan, however, upon completing DBScan, our clusters came out completely unbalanced sizes, which will be later 
explained in results, but in the end we ended up going for KMeans++ and Hierarchical Clustering. 

## [Section 3] 
![image](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/fb251c72-6e4c-4d44-b737-fc9cfb91aac1)

![dendogram](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/d564746e-98a7-416b-b05b-a5fb9680a236)
![ElbowMethod-K](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/8cce210c-ffae-4860-8245-a8d7a3bed195)


## [Section 4] Results  
Our results ended up being cut up into 5 clusters, which clearly makes sense, due our Elbow Method Graph and Dendrogram in Section 3. 
The intial method for analyzing each cluster that we tried to impliment was to look at the correlations of each set of variables and try to infer the data ourselves, 
but since we had 11 columns/variables to worry about, that ended up being simply impossible. The visual that we tried to do that analysis on (PairPlot for K-Means Clustering) can be seen below. 
  
![image](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/a34606ee-f99a-43ca-a1c5-d38f30444799)

Finally, we decided to take the average of every column prior to analysis and then take the average of each column for each cluster created and compare them to one another. 
This meant that we had a baseline value to compare each cluster to and see if some DataSet feature is below average or above average in each cluster. The data collected for K-Means Clustering++ is below: 

### Cluster 0:
### Below average features: 
- accountTotal
- itemCosts
- emergencyFunds
- itemBuyFrequency
- emergencyUseFrequency
- itemCount
- monthlyPaid
- accountType_Premium
- accountType_Student
### Above average features: 
- accountLifespan
- accountType_Regular
***
### Cluster 1:
### Below average features: 
- accountTotal
- itemCosts
- emergencyFunds
- itemBuyFrequency
- emergencyUseFrequency
- itemCount
- monthlyPaid
- accountType_Premium
- accountType_Regular
### Above average features: 
- accountLifespan
- accountType_Student
***
### Cluster 2:
### Below average features: 
- accountType_Regular
### Above average features: 
- accountTotal
- itemCosts
- emergencyFunds
- itemBuyFrequency
- emergencyUseFrequency
- itemCount
- monthlyPaid
- accountLifespan
- accountType_Premium
- accountType_Student
***
### Cluster 3:
### Below average features: 
- accountTotal
- itemCosts
- emergencyFunds
- itemBuyFrequency
- emergencyUseFrequency
- itemCount
- monthlyPaid
- accountType_Regular
- accountType_Student
### Above average features: 
- accountLifespan
- accountType_Premium
***
### Cluster 4:
### Below average features: 
- accountTotal
- itemCosts
- itemBuyFrequency
- itemCount
- monthlyPaid
- accountLifespan
- accountType_Premium
### Above average features: 
- emergencyFunds
- emergencyUseFrequency
- accountType_Regular
- accountType_Student
***
For hierarchical clustering, we decided to use the exact same method and found that the clusters match perfectly, except for cluster 2, where the Below Average value ended up being accountType_Student, instead of accountType_Regular, but that difference is negligeble and actually sense in our analysis. 

Cluster 0: This cluster is characterized by having everything below average, except the amount of time that the account has been open and the amount of customers that hold a regular account. From this, we can infer that these users are pretty inactive, as their account has been open long, but their data is below average. 

Cluster 1: This cluster is characterized the exact same way as cluster 0, but has a greater amount of customers that hold a student account as opposed to a regular account. We could infer that this means the people in this cluster are inactive again, but since the algorithms decided to cut the two into seperate clusters, it means that there's a reason for this and that the predominant population of this cluster is students who probably do not have much money to spend on online shopping and are prone to saving. 

Cluster 2: This cluster is characterized by having everything above average, except accountType_Regular/Student. This means that this clusters spends the most amount of money and buys the most (mostly premium users), so they are very loyal and probably contribute substantialy to the networth and could be considered the elite customers of this company. 

Cluster 3: This cluster is characterized the same as cluster 0, but has above average users with premium accounts. This is a great cluster have, as now ShopEasy can decide to specifically send different types of promotions to inactive users that hold premium accounts, to try to incentivize them to make the best out of their premium account, as opposed to sending the same offers to cluster 0, which do not hold premium account. 

Cluster 4: This cluster holds above average emergencyFunds, emergencyUseFrequency, accountType_Regular and accountType_Student, which to us means that these people stock up on their emergency funds and make very calculated purchases and know exactly what they want. So basically they plan their purchases very well. 

Based on the analysis above, we've decided to name each cluster, as shown below:

<img width="420" alt="Screenshot 2023-12-04 at 15 57 05" src="https://github.com/marekmoch/ShopEasy-286021/assets/151950348/d3586898-26d6-4295-aea2-8ec5a3d455ff">


## [Section 5] Conclusion 
To conclude, we've gained a deeper understanding in implimenting the different clustering techniques, determining what variables, such as the amount clusters or what EPS value to choose, and what data features provide valuable insight and which do not. Additionally, we've learned most of the steps of explanatory data analysis and at what point they should be used. It was also very interesting trying to determine the types of customers framed in each cluster and the methods used to achieve these. As a side-note, we also found that trying to segment a company's customer base accurately is actually a very difficult task and even thought our data was definitely smaller than what an actual company would use to try to segment their customers, we have gained a definite appreciation for those company's and their workers. I know I will definitely not be as mad as I've been up to this point if I recieve a target mail that actually does not make sense at all, since the company just probably just assigned me to the wrong cluster by accident. 

Finally, we are left with multiple different questions, with the main one being if our clusters are actually true. Since we do no a labeled dataset, there is actually no way for us to analyze that (except wit the ways we already did in section 3, which are probably not enough). One possible method or a natural step we could implement in future would (if we were ShopEasy workers), would be to start sending out specialized promotions and targeted ads based on these clusters and then do a survey and see if people are happy with them. Additionaly, this date is static, so we do not know if this is even true for each customer, after we finish doing the clustering algorithm. The next possible steps for that could be to try and implement something that tracks a users data change over time and then tries to create those clusters, but this would be very computationaly intensive. I would also like to take the survey results we would do for each customer targeted experience and include the results in the cluster data, so it could see what cluster it already assigned it to and if the user was happy with it. I think that that would definitely help to improve the clustering model. 
