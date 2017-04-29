library(NLP)
library(tm)

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
subset_index <- sample(c(1:length(review_train[,1])), nrow(review_train))
review_train_subset <- review_train[subset_index,]

reviews <- as.vector(review_train_subset$text)

# Some custom cleaning function we might consider to add
f <- function(x, pattern){gsub(pattern, "", x)}
custom_transformation <- content_transformer(f)
# tm_map(review_corpus, custom_transformation, "[[:digit:]]+")

cleanCorpus = function(corpus){
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

# Clean reviews using the above function
review_corpus <- cleanCorpus(Corpus(VectorSource(reviews)))

# Change type to DocumentTermMatrix
review_dtm <- DocumentTermMatrix(review_corpus)
dim(review_dtm) #6679

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
cleaned_review_dtm <- removeSparseTerms(review_dtm, 0.99)
dim(cleaned_review_dtm) #759
# inspect(cleaned_review_dtm_train[1:5,1:10])
# cleaned_review_dtm_train
# Do not run the following line if you computer do not have enough memory...
# I trained a subset of review data of 10000 observations, which also failed.
# Should use a much smaller subset.
cleaned_review_dtm = data.frame(as.matrix(cleaned_review_dtm))
cnames = colnames(cleaned_review_dtm)
X = cbind(cleaned_review_dtm, review_train_subset$stars)
colnames(X) = c(cnames, "label")
X_train = X[1:100000,]
X_valid = X[100001:116474,]


lin_mod = glm(label~., data = X_train)
summary(lin_mod)
pred =predict(lin_mod, newdata = data.frame(X_valid))
yulaoban = function(x){
  min(5, max(1, x))
}
pred = sapply(pred, yulaoban)
mean(( pred - X_valid$label)^2)




