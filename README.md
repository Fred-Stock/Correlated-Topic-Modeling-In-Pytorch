# Correlated-Topic-Modeling-In-Pytorch

Develop a PyTorch model to implement Corrlelated Topic Models (CTM).

Original paper - [Correlated Topic Models](https://proceedings.neurips.cc/paper_files/paper/2005/file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf)

A simple blog post explaining CTM - [Intuitive Guide to Correlated Topic Models](https://towardsdatascience.com/intuitive-guide-to-correlated-topic-models-76d5baef03d3)

References:
1. [Latent Dirichlet Allocation LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation#:~:text=In%20natural%20language%20processing%2C%20Latent,of%20a%20Bayesian%20topic%20model.)
2. Python implementation of CTM: [TomotoPy](https://github.com/bab2min/tomotopy)
3. [PyCTM](https://github.com/kzhai/PyCTM)
4. Implementation of Topic Models in [R](https://rpubs.com/chelseyhill/672546)
5. R documentation of [topicmodels](https://cran.r-project.org/web/packages/topicmodels/index.html)

Members:
1. Fred Stock
2. Anugrah Vaishnav
3. Nuno Mestre
4. Flore Kenne
5. Shafaat Osmani

Running Our Code:
ctm_model.py - This is our implementation of a correlated topic model. The main function is at the bottom of
the file. The function RunModel() takes three parameters: the set of documents to analyze (in a csv), the the number of items from that set to use, and the number of topics to find in the corpus. Simply set the parameters as you wish and then run the Python file. 

In ./Other Model Implementations you will find code that will run a pre-existing CTM implementation. There are two files, one for topic-models an R library, and tomotopy, a cpp library with a python front end. Both of these files can be run as they are without modification, or the parameters in the CTM() and RunModel() functions, respectively, can be modified to run on different parameters. 