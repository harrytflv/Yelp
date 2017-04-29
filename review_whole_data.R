library(NLP)
library(tm)
setwd("~/154final/Yelp")

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
review_test <- read.csv('yelp_academic_dataset_review_test.csv')
whole_data <- rbind(review_train[,c(1:6,8:11)], review_test)
reviews <- as.vector(whole_data$text)
# dim(whole_data)
# dim(review_test)

stars <- c(review_train[,7], rep(0,21919))

cleanCorpus = function(corpus){
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

# Clean reviews and store them as corpus
review_corpus <- cleanCorpus(Corpus(VectorSource(reviews)))

# Split reviews into single tokens
review_df <- data.frame(text=sapply(review_corpus, identity), stringsAsFactors=F)
splited_review <- sapply(review_df , function(x) strsplit(x, " "))

# Grab ngrams
nGram <- function(tokens, num_grams){
  return(vapply(ngrams(tokens, num_grams), paste, "", collapse = " "))
}

bi_grams <- lapply(splited_review, function(x, y) nGram(x, 2L))

# Paste bigram together, so we can store the data as DocumentTermMatrix
removeSpace <- function(token){
  return(sapply(token, function(x)gsub(" ", "",x) ))
}
pasted_bi_grams <- lapply(bi_grams, removeSpace)
pasted_bi_grams_corpus = Corpus(VectorSource(pasted_bi_grams))
bi_grams_dtm = DocumentTermMatrix(pasted_bi_grams_corpus)
dim(bi_grams_dtm)
# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
bi_grams_dtm_9996 = removeSparseTerms(bi_grams_dtm, 0.9996)
# dim(bi_grams_dtm_9996)

# Create a ngram matrix
cleaned_review_dtm = data.frame(as.matrix(bi_grams_dtm_9996))
cnames = colnames(cleaned_review_dtm)
X = cbind(cleaned_review_dtm, stars)
colnames(X) = c(cnames, "stars")

write.csv(X, file = "reviewTrainAndTest.csv")
# Split data into training and testing
X_train = X[1:116474,]
X_test = X[116474:138393,]

# Fit linear model
lin_mod = glm(stars~., data = X_train)
# summary(lin_mod)
# a <- summary(lin_mod)$coef
pred = predict(lin_mod, newdata = data.frame(X_test))
yulaoban = function(x){
  min(5, max(1, x))
}

business_id = X_test$business_id 
pred = sapply(pred, yulaoban)

reviewTestOutPut = cbind(business_id, pred)
colnames(reviewTestOutPut) = c("business_id","stars")

write.csv(reviewOutPut, file = "reviewTestOutPut.csv")