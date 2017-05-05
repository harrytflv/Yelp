library(NLP)
library(tm)

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
review_test <- read.csv('yelp_academic_dataset_review_test.csv')
# combine train data set and test data together
whole_data <- rbind(review_train[,c(1:6,8:11)], review_test)
reviews <- as.vector(whole_data$text)
stars <- c(review_train[,7], rep(0,21919))

# corpus function git hint from Raj.
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

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
cleaned_review_dtm <- removeSparseTerms(review_dtm, 0.99)
cleaned_review_dtm = data.frame(as.matrix(cleaned_review_dtm))
cnames = colnames(cleaned_review_dtm)

# combine the clean data with stars
X = cbind(cleaned_review_dtm, stars)
colnames(X) = c(cnames, "review_stars")

# write the clean wordbag data to a CSV
write.csv(X, "wordbagCleanTrainAndTest.csv")
haha <- read.csv("wordbagCleanTrainAndTest.csv")

# split it to two part as REAL TRAIN and REAL TEST 
X_train = X[1:116474,]
X_test = X[116475: nrow(X),]


# Use linear model to predict the review test based on 
# wordbagCleanTrainAndTest.csv
lin_mod = glm(review_stars~., data = X_train)
pred =predict(lin_mod, newdata = data.frame(X_test))
# since the value may not in (1,5), so we need to write a function to void that.
yulaoban = function(x){
  min(5, max(1, x))
}
# apply the function to the pred
pred = sapply(pred, yulaoban)

# combine the business_id and stars together, then write it out for Kaggle test.
business_id = as.vector(review_test$business_id)
wordbag_output = cbind(business_id, pred)
colnames(wordbag_output) = c("business_id", "stars")
write.csv(wordbag_output,"xuran_wordbag_output.csv")




