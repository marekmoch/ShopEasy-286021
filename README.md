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

Our design choices aren't really that complicated. We tried to make everything as readable as possible. We kept most of the default colors, but went ahead and changed some, such as when we displayed the red dashed line to show our only missing row value for one of the features. For our ipynb file, we tried to follow a chronological order, like if a person were following a story. 
   
Preprocessing the dataset was an easy task. The personId column is a unique identifier, so every row is "unique" in its own sense, but we 
still ended up checking for duplicate personId values just to be safe (there weren't any). We also decided to encode categorical features at 
the beginning of EDA since they also offer an insight into analyzing NULL values and the general distribution of the data, so preprocessing
was pretty much already finished.  

---

Below are all of the attributes we decided to drop before training our data and the reasons why:   

#### Location Attributes (New York, Los Angeles, Chicago):
Redundancy and Relevance: In an online shopping context, the physical location of a user is less relevant, especially if the platform's services and offerings are the same across regions.   
Lack of Correlation: The heatmap analysis indicated no significant correlation between these location attributes and other features, suggesting they wouldn't contribute meaningful variance or distinction in customer segmentation.

#### WebUsage and FrequencyIndex:
Redundancy with Item Buy Frequency: These features would overlap with itemBuyFrequency in terms of the information they provide, as they all keep track of how often the users shop on the e-commerce platform. As itemBuyFrequency adequately represents the shopping frequency, including additional similar measures would introduce redundancy and dilute the distinctiveness of clusters. Therefore we decided to simplify it and remove these redundant features to reduce the dimensionality and improve interpretability of the clusters. 

#### SingleItemBuyFrequency and MultipleItemBuyFrequency:
Overlap with General Purchase Frequency: The distinction between single and multiple item purchases would not significantly contribute to understanding different customer behaviors, especially as the overall purchase frequency (itemBuyFrequency) is already considered.      
Focus on Overall Behavior: In our clustering , we aim to identify broader patterns and behaviors, and the distinction between single and multiple item purchases would be too detail-oriented for our clustering purposes.

#### SingleItemCost and MultipleItemCost:
Incorporated into Total Item Cost: The total cost of items purchased (itemCosts) captures the essential spending behavior of customers. Distinguishing between single and multiple item costs would not add meaningful insights for clustering.Total spending will be more relevant than the composition of that spending.

#### LeastAmountPaid:
Limited Insight: The minimum amount a customer has paid in a transaction would not provide significant insights into their overall shopping behavior or value to the platform.

#### MaxSpendLimit:
Derived Attribute: As it is simply a derivative of user behavior and loyalty, this measurement would not be as informative as direct measures of these attributes through the other selected features.      
Unclear Calculation Method: Without understanding how this limit is calculated, relying on it for segmentation might lead to misleading conclusions.

#### PaymentCompletionRate:
Irrelevance for Segmentation: The rate of completing payments does not differentiate customer segments in a meaningful way for the purpose of understanding shopping habits or preferences. Looking for example at total spending and purchase frequency instead, will be more insightful.

#### EmergencyCount:
Overlap with Emergency Use Frequency: Since emergencyUseFrequency already provides a measure of how frequently customers use their emergency funds, emergencyCount would be redundant, which is shown by their correlation of 0.8.   
Frequency vs. Count: The frequency of using emergency funds is likely a more relevant metric for understanding customer behavior than just the count of such instances.

Below are all of the attributes we decided to retain before training our data and the reasons why: 

The features we have chosen to keep provide a comprehensive view of customer behavior on the platform, focusing on their spending habits, purchase frequencies, and overall engagement with the platform. By eliminating redundant or less informative features, we enhance the potential for the clustering algorithm to allow us to discover meaningful and distinct customer segments based on the most impactful aspects of their behavior. Additionaly, the reason we kept the account type (Premium, Student or Standard), even though there seemed to be no correlation between anything and makes sense to delete it, is because, as we already said, Shopeasy wants to "target specific customer segments for their promotional efforts". So we believe that it is very important for Shopeasy to know whether their customer base are primarily students, standard or premium holders because that way they can target their audience more. Even though this design choice might decrease the silhoutte score by a lot and might seem illogical, we believe that in the real world this would make a lot of sense and would be definitely more beneficial. 

---
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
Below are the experiments that we conducted to validate all of our target contributions. 
#### K-Clustering++ Elbow-Method:
![ElbowMethod-K](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/8cce210c-ffae-4860-8245-a8d7a3bed195)
   
K-Means: We used the elbow method to discover what the most appropriate amount of clusters are (see below). The Inertia, which is plotted on the y-axis, measures how well a dataset was clustered by K-Means for each amount of clusters (x-axis). It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster. A good model is one with low inertia and a low number of clusters. You find the most appropriate amount of clusters by finding the “elbow” in the graph. For our project, the elbow method suggested that it is best to use 5 distinct clusters, which is  why we settled on this amount. 
   
#### DBSCAN Elbow Method:
![image](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/fb251c72-6e4c-4d44-b737-fc9cfb91aac1)

Similar to K-Means, the elbow method can be used for this method as well. In this case it is to tune the hyper parameter eps, which is an input value which defines the maximum distance between two samples for them to be considered as in the same neighborhood; essentially how close points should be to each other to be considered part of a cluster. In the graph (see below),  the elbow is at the point where the curve starts to ascent rapidly. Selecting an eps value just before this sharp increase gives us the best clustering results, as it balances the density requirement for clusters without merging separate clusters or including too much noise. We have located it to be around the value of 1.8. 

#### Hierarchical Clustering Cluster Amount:
![image](https://github.com/marekmoch/ShopEasy-286021/assets/151950348/aa364c59-72df-4a4d-bbd1-0501ae4b43fd)
   
The image above is a dendrogram, which shows how hierarchical clustering groups the different clusters together. As can be seen, the biggest apart distance is create when around five clusters are made, therefore that will our cut off point for hierarchical clustering. 
   
#### Silhouette Score:
To confirm the clusters we have determined using the various clustering techniques, we computed the Silhouette Score score for each of them. This score is a measurement of how well samples are clustered with samples that are similar to themselves. It measures how similar an object is to its own cluster compared to other clusters. The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. Depending on the application, more distinct (higher value) or less distinct (lower value) will be most fitting. 
   
The results were:
K-means: ​​0.33334945234618624
DB-Scan: 0.35529034595145237
Hierarchical Clustering: 0.3248860421103077
   
The approximate silhouette score of 0.3 suggests that while there is some overlap between clusters, each cluster has distinct characteristics that are meaningful for our objectives. A score of 0.3, which is a moderate score, in this case, indicates naturally overlapping clusters. We are creating clusters which try to segment customer behaviors and it's therefore natural that some overlap will naturally take place. The clusters align well with our goals and provide meaningful insights, as we will discuss in the next section. A silhouette score that is not very close to +1 can therefore still be considered successful.It indicates that while there is some overlap, the clusters are still distinct to provide information for the EasyShop business to act upon. Also, we did say that we assume that our silhoutte score might be lower, due to our choice, but that does make sense. We now hope to see a clear distinction between the clusters, detailing premiumm, standard and student accounts. 

#### Empirical Experimentation:
In addition to the abovementioned formal and streamlined  experiments, we also experimented by using a wide range of different hyperparameters as inputs for the clustering functions. Also, we tried running the clustering algorithms with varying column attributes as inputs. We finally settled on the features we described above. 

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
To conclude, we have gained a depper understanding in implementing clustering mathods based on machine learning. We have learned the steps of Explanatory Data Analysis and how to preprocess the data for the clustering algorithms. We have enquired how to impliment different clustering techniques, determining which variables and hyperparameters, such as the amount clusters or the EPS value, to choose, as well as which data features provide valuable insight and which do not. It was also very interesting trying to determine the types of customers framed in each cluster and how different clustering techniques will identify non-identical clusters. We are very satisfied with the clusters we have found, since they clearly map onto different customer behaviour types, which would allow Shopeasy to target their customers significantly better and provide a better user experience. As a side-note, we also found that trying to segment a company's customer base accurately is actually a very challenging task and even though our data was definitely smaller than what an actual company would use to try to segment their customers, we have gained a definite appreciation for these companies and their workers. We know we will definitely not be as confused as we have been up to this point if we recieve a targeted mail that actually does not make any sense to us at all, since the company probably just assigned us to the wrong cluster by accident. 

Finally, we are left with multiple different questions, the main one being, if our clusters are actually accurate. Since we dont have a labeled dataset, there is no way for us to analyze that directly, except with methods such as the ones we used in section 3. In a real world scenario, one possible method or a natural step we could implement in future (if we were ShopEasy workers), would be to start sending out user-specific  promotions and targeted ads based on these clusters and then track user behaviour or survey the customers to evaluate the cluster accuracy. We could then use this data as additional input in the clustering methods to further increase the accuracy of the clusters. Additionaly, we aknowledge that a customer's behaviour might change over time. Our data is static and therefore we are not able to assess if a customer should be swapped to a different cluster, once we have finished running the clustering algorithm. Hence, the next step to combat this limitation would be to implement something that tracks users' data change over time and then tries to update the clusters. This would be computationaly intensive, but it would likely be beneficial for the E-Commerce business. These additional changes would definitely help to improve the clustering model further.
