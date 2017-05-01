library(NLP)
library(tm)

# Read data and check dimension
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
subset_index <- sample(c(1:length(review_train[,1])), nrow(review_train))
review_train_subset <- review_train[subset_index,]
reviews <- as.vector(review_train_subset$text)
stars <- review_train_subset[,7]

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
  return(sapply(token, function(x) gsub(" ", "",x) ))
}
pasted_bi_grams <- lapply(bi_grams, removeSpace)
pasted_bi_grams_corpus = Corpus(VectorSource(pasted_bi_grams))
bi_grams_dtm = DocumentTermMatrix(pasted_bi_grams_corpus)

# Set the frequency threshold to be 0.99, so we can choose proper features to
# decrease sparsity.
bi_grams_dtm = removeSparseTerms(bi_grams_dtm, 0.99)

# Create a ngram matrix
cleaned_review_dtm = data.frame(as.matrix(bi_grams_dtm))
cnames = colnames(cleaned_review_dtm)
X = cbind(cleaned_review_dtm, stars)
colnames(X) = c(cnames, "label")

# Split data into training and testing
X_train = X[1:100000,]
X_valid = X[100001:116474,]

# Fit linear model
lin_mod = glm(label~., data = X_train)
summary(lin_mod)
pred =predict(lin_mod, newdata = data.frame(X_valid))
yulaoban = function(x){
  min(5, max(1, x))
}
pred = sapply(pred, yulaoban)
mean((pred - X_valid$label)^2)

