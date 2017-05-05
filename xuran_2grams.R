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

# Clean reviews and store them as corpus
review_corpus <- cleanCorpus(Corpus(VectorSource(reviews)))

# Split reviews into single tokens
review_df <- data.frame(text=sapply(review_corpus, identity), stringsAsFactors=F)
splited_review <- sapply(review_df , function(x) strsplit(x, " "))

# Grab ngrams
nGram <- function(tokens, num_grams){
  return(vapply(ngrams(tokens, num_grams), paste, "", collapse = " "))
}


# 2 grams and 3grams
bi_grams <- lapply(splited_review, function(x, y) nGram(x, 2L))
# tri_grams <- lapply(splited_review, function(x, y) nGram(x, 3L))
# Paste bigram together, so we can store the data as DocumentTermMatrix
removeSpace <- function(token){
  return(sapply(token, function(x)gsub(" ", "",x) ))
}
pasted_bi_grams <- lapply(bi_grams, removeSpace)
# pasted_tri_grams <- lapply(tri_grams, removeSpace)
pasted_bi_grams_corpus = Corpus(VectorSource(pasted_bi_grams))
#pasted_tri_grams_corpus = Corpus(VectorSource(pasted_tri_grams))
bi_grams_dtm = DocumentTermMatrix(pasted_bi_grams_corpus)
# tri_grams_dtm = DocumentTermMatrix(pasted_tri_grams_corpus)


# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
bi_grams_dtm_9975 = removeSparseTerms(bi_grams_dtm, 0.9975)
# tri_grams_dtm_9995 = removeSparseTerms(tri_grams_dtm, 0.9995)

# Create a ngram matrix
cleaned_review_dtm = data.frame(as.matrix(bi_grams_dtm_9975))
# cleaned_review_dtm = data.frame(as.matrix(tri_grams_dtm_9995))

cnames = colnames(cleaned_review_dtm)
X = cbind(cleaned_review_dtm, stars)
colnames(X) = c(cnames, "review_stars")

write.csv(X, file = "review_3grams_TrainAndTest.csv")
# Split data into training and testing
X_train = X[1:116474,]
X_test = X[116475:138393,]

# Fit linear model
lin_mod = glm(review_stars~., data = X_train)
pred = predict(lin_mod, newdata = data.frame(X_test))

yulaoban = function(x){
  min(5, max(1, x))
}

pred = sapply(pred, yulaoban)
xuran_2grams_output = cbind(as.vector(review_test$business_id), pred)
colnames(xuran_2grams_output) = c("business_id", "stars")
write.csv(xuran_2grams_output, file = "xuran_2grams_output.csv")
# write.csv(xuran_2grams_output, file = "xuran_3grams_output.csv")

