setwd("~/154data")

review_train <- read.csv('yelp_academic_dataset_review_train.csv')
bus_train <- read.csv('yelp_academic_dataset_business_train.csv')

library(clustMixType)
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
bus_train <- read.csv('yelp_academic_dataset_business_train.csv')
extended_bus_train <- read.csv('extended_business_train.csv')

extended_bus_train <- extended_bus_train[,c(-1,-2,-7,-8,-9)]
extended_bus_train <- extended_bus_train[,c(-3,-7)]
extended_bus_train_factors <- extended_bus_train[,c(1,3,4,7,12:length(extended_bus_train))]
extended_bus_train <- extended_bus_train[,-c(1,3,4,7,12:length(extended_bus_train))]
extended_bus_train <- extended_bus_train[,c(1,4,3,5)]

extended_bus_train_factors[] <- lapply(extended_bus_train_factors, as.character)
extended_bus_train_factors[is.na(extended_bus_train_factors)] <- "NNN"
extended_bus_train_factors[] <- lapply(extended_bus_train_factors, as.factor)

cleaned_data <- cbind(extended_bus_train,extended_bus_train_factors)
class(cleaned_data)

clusters <- kproto(cleaned_data[,3:length(cleaned_data)], k = 10)
kpres <- kproto(c, 2)
clprofiles(clusters, cleaned_data)

clusters$cluster





review_test <- read.csv('yelp_academic_dataset_review_test.csv')
bus_test <- read.csv('yelp_academic_dataset_business_test.csv')


# Put all data together
whole_data <- merge(review_train, bus_train, by = "business_id")

# Clean business train data
review_train <- review_train[,c("user_id","text","business_id","useful",
                                "cool","funny","stars")]
extended_bus_train <- extended_bus_train

# Put all data together
# whole_data <- merge(review_train, bus_train, by = "business_id")
whole_data <- merge(review_train, extended_bus_train, by = "business_id")

# Change factors to vectors, and replace NA with "NA"

clusters <- kmeans(na.omit(whole_data), centers = 10)$cluster
# clusters <- kmeans(c(extended_bus_train$longitude, na.omit(extended_bus_train$neighborhood))
                   # , centers = 5)$cluster
# whole_data[] <- lapply(whole_data, as.character)
# whole_data[is.na(whole_data)] <- 999
# whole_data$Alcohol
# clusters <- kmeans(whole_data$Alcohol, centers = 10)$cluster
# clusters <- kmeans(c(1,2,3,4,5,6,7,8,9,0), centers = 3)$cluster
# clusters <- kmeans(data_before_grouping[,5:6], centers = 2000)$cluster

clusters <- kmeans(na.omit(whole_data), centers = 10)$cluster
# na.omit(whole_data)


useful_cols <- c("user_id","business_id","text","city","longitude","latitude","stars.x")
data_before_grouping <- whole_data[,useful_cols]


clusters <- kmeans(data_before_grouping[,5:6], centers = 2000)$cluster
grouped_data <- cbind(data_before_grouping, clusters)
cluster5 <- grouped_data[grouped_data$clusters == 1034, c("user_id","business_id","stars.x")]

user_reviews_matrix = spread(data = cluster5, key = business_id, value = stars.x, fill = 0)

plot(grouped_data$latitude, grouped_data$longitude)

a <- tapply(grouped_data$business_id, grouped_data$clusters, length)
a
max(a)
which.max(a)
grouped_data[grouped_data$clusters == 993, c("user_id","business_id","stars.y")]


hist(clusters)


grouped_data <- cbind(data_before_grouping, clusters)
cluster5 <- grouped_data[grouped_data$clusters == 1034, c("user_id","business_id","stars.x")]
user_reviews_matrix = spread(data = cluster5, key = business_id, value = stars.x, fill = 0)


grouped_data[grouped_data$clusters == 993, c("user_id","business_id","stars.y")]


