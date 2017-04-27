library(NLP)
library(tm)
setwd("~/154")


# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
dim(review_train)
subset_index <- sample(c(1:length(review_train[,1])), 1000)
review_train_subset <- review_train[subset_index,]

reviews <- as.vector(review_train_subset$text)


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
review_corpus <- cleanCorpus(Corpus(VectorSource(reviews)))
Corpus(VectorSource(reviews))
inspect(review_corpus[1])

# Change type to DocumentTermMatrix
review_dtm <- DocumentTermMatrix(review_corpus)
dim(review_dtm) #6679

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
cleaned_review_dtm <- removeSparseTerms(review_dtm, 0.99)
dim(cleaned_review_dtm) #759
inspect(cleaned_review_dtm[1:5,1:10])

# Do not run the following line if you computer do not have enough memory...
# I trained a subset of review data of 10000 observations, which also failed.
# Should use a much smaller subset.
X_train = as.matrix(cleaned_review_dtm)
y_train = review_train_subset$stars
lin_mod = lm(y_train ~ X_train)
summary(lin_mod)
