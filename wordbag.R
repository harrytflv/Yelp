library(NLP)
library(tm)
setwd("~/154data")

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
dim(review_train)
subset_index <- sample(c(1:length(review_train[,1])), 1200)

subset_index_train <- subset_index[1:1000]
subset_index_valid <- subset_index[1001:1200] 

review_train_subset_train <- review_train[subset_index_train,]
review_train_subset_valid <- review_train[subset_index_valid,]

reviews_train <- as.vector(review_train_subset_train$text)
reviews_valid <- as.vector(review_train_subset_valid$text)

# Some custom cleaning function we might consider to add
f <- function(x, pattern){gsub(pattern, "", x)}
custom_transformation <- content_transformer(f)
tm_map(review_corpus, custom_transformation, "[[:digit:]]+")

cleanCorpus = function(corpus){
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

# Clean reviews using the above function
review_corpus_train <- cleanCorpus(Corpus(VectorSource(reviews_train)))
Corpus(VectorSource(reviews_train))
inspect(review_corpus_train[1])

# Change type to DocumentTermMatrix
review_dtm_train <- DocumentTermMatrix(review_corpus_train)
dim(review_dtm_train) #6679

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
cleaned_review_dtm_train <- removeSparseTerms(review_dtm_train, 0.99)
dim(cleaned_review_dtm_train) #759
inspect(cleaned_review_dtm_train[1:5,1:10])
cleaned_review_dtm_train
# Do not run the following line if you computer do not have enough memory...
# I trained a subset of review data of 10000 observations, which also failed.
# Should use a much smaller subset.
X_train = as.matrix(cleaned_review_dtm_train)
y_train = review_train_subset$stars


# Clean reviews using the above function
review_corpus_valid <- cleanCorpus(Corpus(VectorSource(reviews_valid)))
Corpus(VectorSource(reviews_valid))
inspect(review_corpus_valid[1])

# Change type to DocumentTermMatrix
review_dtm_valid <- DocumentTermMatrix(review_corpus_valid)
dim(review_dtm_valid) #6679

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
cleaned_review_dtm_valid <- removeSparseTerms(review_dtm_valid, 0.99)
dim(cleaned_review_dtm_valid) #759
inspect(cleaned_review_dtm_valid[1:5,1:10])
cleaned_review_dtm_valid
# Do not run the following line if you computer do not have enough memory...
# I trained a subset of review data of 10000 observations, which also failed.
# Should use a much smaller subset.
X_valid = as.matrix(cleaned_review_dtm_valid)
y_valid = review_train_subset_valid$stars




lin_mod = glm(y_train ~ X_train)
summary(lin_mod)
pred =predict(lin_mod, newdata = data.frame(X_valid))
mean((pred - y_valid)^2)
