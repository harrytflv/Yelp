library(NLP)
library(tm)

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
review_test <- read.csv('yelp_academic_dataset_review_test.csv')
# dim(review_test)
whole_data <- rbind(review_train[,c(1:6,8:11)], review_test)
reviews <- as.vector(whole_data$text)
# dim(whole_data)
# dim(review_test)
stars <- c(review_train[,7], rep(0,21919))
max(stars)

# Some custom cleaning function we might consider to add
# f <- function(x, pattern){gsub(pattern, "", x)}
# custom_transformation <- content_transformer(f)
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
# max(stars)

X = cbind(cleaned_review_dtm, stars)
colnames(X) = c(cnames, "review_stars")
# max(X$review_stars)
# stars == X$review_stars

write.csv(X, "wordbagCleanTrainAndTest.csv")
haha <- read.csv("wordbagCleanTrainAndTest.csv")
# View(haha)
# View(haha["review_stars"])
# min(haha["review_stars"])
# max(haha["review_stars"])
X_train = X[1:116474,]
# dim(X_train)

X_test = X[116475: nrow(X),]
# dim(X_test)


lin_mod = glm(review_stars~., data = X_train)
# summary(lin_mod)
pred =predict(lin_mod, newdata = data.frame(X_test))
yulaoban = function(x){
  min(5, max(1, x))
}
pred = sapply(pred, yulaoban)

business_id = as.vector(review_test$business_id)
wordbag_output = cbind(business_id, pred)
colnames(wordbag_output) = c("business_id", "stars")

write.csv(wordbag_output,"xuran_wordbag_output.csv")




