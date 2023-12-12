# This code utilizes the library mentioned in the paper topicmodels: An R Package for Fitting Topic Models

# Load Necessary Libraries
library(tm)
library(topicmodels)
library(corrplot)

# Load the Data
data <- read.csv("./newsgroups_data.csv")
docs <- as.character(data$content)

# Select only the first 20 documents
docs <- docs[1:20]

# Data Preprocessing
corpus <- Corpus(VectorSource(docs))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)

# Fit the Correlated Topic Model (CTM)
num_topics <- 5  # Number of topics set to 5
ctm_model <- CTM(dtm, k = num_topics, method = "CTM")

# Extracting the topic-term matrix
topic_terms <- terms(ctm_model, 6)
print(topic_terms)

# Extract and Print the Correlation Matrix
correlation_matrix <- ctm_model@Sigma
correlation_matrix_normalized <- cov2cor(correlation_matrix)
print(correlation_matrix)

# Visualizing the Normalized Correlation Matrix with Numbers
corrplot(correlation_matrix_normalized, method = "color", addCoef.col = "black", is.corr = TRUE)
