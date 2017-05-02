setwd("~/154data")
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
bus_train <- read.csv('yelp_academic_dataset_business_train.csv')

review_test <- read.csv('yelp_academic_dataset_review_test.csv')
bus_test <- read.csv('yelp_academic_dataset_business_test.csv')

# Put all data together
whole_data <- merge(review_train, bus_train, by = "business_id")

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
