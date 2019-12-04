# VTP-Customer-Segmentation
Inspired by the infamous RFM (Recency, Frequency, Monetary) segmentation framework in Marketing, 
I decide to built an end to end machine learning model used for customer segmentation for Viettel Pay. 
This repository contains the dataset, Feature Engineering and cleaning notebook, Clustering Notebook, Visualization Notebook and Pipeline.
 
The main algorithm of choice is K-Prototype due to its robust capability to handle both numerical and categorical nature of the dataset.
This property is due to a special mix of distance metric smoothly (or not) combining both Euclidean distance (KMeans) and Hamming distance (K-Modes)
For further information,  refer to Huang(97)


DISCLAIMER: for security reasons, only a small subset of the dataset will be uploaded and this model is not the final version 
ready for deployment

REFERENCES
Huang, Z.: Clustering large data sets with mixed numeric and categorical values, Proceedings of the First Pacific Asia Knowledge Discovery and Data Mining Conference, Singapore, pp. 21-34, 1997.
