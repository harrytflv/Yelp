setwd("~/154data")
review_train <- read.csv('yelp_academic_dataset_review_train.csv')
bus_train <- read.csv('yelp_academic_dataset_business_train.csv')

review_test <- read.csv('yelp_academic_dataset_review_test.csv')
bus_test <- read.csv('yelp_academic_dataset_business_test.csv')

whole_data <- merge(review_train, bus_train, by = "business_id")
plot(whole_data$city, whole_data$stars.y)
plot(whole_data$city)
useful_cols <- c("business_id","text","longitude","latitude","stars.y")
data_before_grouping <- whole_data[,useful_cols]
grouped_data <- kmeans(whole_data, centers = 100)
