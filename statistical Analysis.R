Qn 1
# t test method
# Given data
mean_class <- 1.8
sd_class <- 1.03
n_class <- 45

mean_gym <- 2.5
sd_gym <- 1.33
n_gym <- 62

set.seed(123)  
data_class <- rnorm(n_class, mean_class, sd_class)
data_gym <- rnorm(n_gym, mean_gym, sd_gym)

result <- t.test(data_class, data_gym, var.equal = FALSE)

print(result)

------------------
  # manual method 
  # Define the data
  mean1 <- 1.8
mean2 <- 2.5
std_dev1 <- 1.03
std_dev2 <- 1.33
n1 <- 45
n2 <- 62

# Calculate standard errors
SE1 <- std_dev1 / sqrt(n1)
SE2 <- std_dev2 / sqrt(n2)

# Calculate degrees of freedom
df <- ((std_dev1^2 / n1 + std_dev2^2 / n2)^2) / 
  ((std_dev1^4 / (n1^2 * (n1 - 1))) + (std_dev2^4 / (n2^2 * (n2 - 1))))

# Calculate t-statistic
t_statistic <- (mean1 - mean2) / sqrt(SE1^2 + SE2^2)

# Calculate p-value
p_value <- 2 * pt(-abs(t_statistic), df = df)
# Calculate p-value
p_value <- 2 * pt(-abs(t_statistic), df = df)

# Print the corrected p-value
cat("p-value:", p_value, "\n")

# Print the results
cat("t-statistic:", t_statistic, "\n")
cat("p-value:", p_value, "\n")

#qn 3 

cystfibr<-"C:/Users/elangovan.paramasiva/Desktop/DS/sem 2/KL7012_ Stastitical/dataqn3.txt"
cystfibr_df <- read.table(cystfibr, header = TRUE)
View(cystfibr_df) 
print(cystfibr_df)

attach(cystfibr_df)
print(age)

# without $ symbol, easily access column by their names alone
detach(cystfibr_df)

summary(cystfibr_df)  

library(ggplot2)

#qn 4 - a


# Define the variables of interest
cys_feature <- c("age","height", "weight", "bmp", "fev1", "rv", "frc", "tlc", "pemax")
pairs(cystfibr_df[, cys_feature], 
      main = "Scatterplot Matrix", 
      pch = 16,  
      col = "blue",
      cex = 1.5,  
      lower.panel = NULL) 

library(corrplot)
print(cor_matrix)
cor_matrix <- cor(cystfibr_df[, c(cys_feature)])
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

#qn 4 - b

# Renaming 0 as male and 1 as female 
cystfibr_df$sex <- factor(cystfibr_df$sex, labels = c("Male", "Female"))
# creating again another feature name column in order of having outlier at top and no outlier at last
cystfibr_bxpt <- c( "height", "weight", "fev1","rv","frc","bmp", "tlc", "pemax")
par(mfrow = c(2, 5))  

for (variable in cystfibr_bxpt) {
  boxplot(cystfibr_df[[variable]] ~ cystfibr_df$sex,
          main = paste("Boxplot of", variable, "by sex"),
          xlab = "Sex",
          ylab = variable,
          outcol = "red",        
          outpch = 16)
}

for (variable in cystfibr_bxpt) {
  boxplot(cystfibr_df[[variable]] ~ cystfibr_df$sex,
          main = paste("Boxplot of", variable, "by sex"),
          xlab = "Sex",
          ylab = variable,
          # Specify colors for box, median, whiskers, and outliers
          boxcol = "blue",          # Box color
          medcol = "orange",           # Median line color
          whiskcol = "green",       # Whiskers color
          outcol = "red",        # Outliers color
          outpch = 16)              # Symbol for outliers
}


# qn 5 

y<-dbinom(5,8,0.87)
y

x <- 5  
n <- 8  
p <- 0.87
q<- 1-p 
y<-dbinom(x,n,p)
y
y<-dbinom(5,8,0.87)
y



#qn 6

# Define parameters
lbd <- 6 
k <- 8
prb_8_email <- dpois(k, lbd)
print(prb_8_email)


#qn 7

# Define parameters
mean_lit <- 14600 
SD_lit <- 2600  
X_lit <- 10000  
Z <- (X_lit - mean_lit) / SD_lit

prob_X_lit <- 1 - pnorm(Z)
print(prob_X_lit)


#qn 8 a

temp <- c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
conv_sugar <- c(8.1, 7.8, 8.5, 9.8, 9.5, 8.9, 8.6, 10.2, 9.3, 9.2, 10.5)

sug_df <- data.frame(temp, conv_sugar)

lin_model <- lm(conv_sugar ~ temp, data = sug_df)

plot(temp, conv_sugar, pch = 16, col = "blue",main = "Linear Regression Line for Temp VS Converted Sugar", xlab = "Temperature", ylab = "Converted Sugar")
abline(lin_model, col = "red",lwd = 2)

summary(lin_model)

#Qn9

adv <- c(0, 10, 4, 5, 2, 7, 3, 6)
purs <- c(4, 12, 5, 10, 1, 3, 4, 8)

correlation <- cor(adv, purs)
print(correlation)

cor_test <- cor.test(adv, purs)
print(cor_test)


add_df <- data.frame(adv, purs)

add_model <- lm(adv ~ purs, data = add_df)
par(mfrow = c(1, 2))  
plot(adv, purs, pch = 16, col = "blue",main = "Linear Regression Line for Advertisement VS Purchase", xlab = "Advertisement", ylab = "Purchase")
abline(add_model, col = "red",lwd = 2)

summary(add_model)

#qn 10

library(visdat)
library(dplyr)
library(ggplot2)
library(tidyr)


traffic_data1 <- read.csv("C:/Users/elangovan.paramasiva/Downloads/qn10updated.csv")
dim(traffic_data1) 
str(traffic_data1 )

vis_miss(traffic_data1)

unique(traffic_data1$Time)

#conv time into catg as it is easier for comparison
traffic_data1 <- traffic_data1[traffic_data1$Time != "", ]
traffic_data1$Timenew <- as.POSIXct(traffic_data1$Time, format = "%I:%M %p")
traffic_data1$Hour <- as.numeric(format(traffic_data1$Timenew, "%H"))
time_intervals <- c(0, 6, 12, 18, 24)

traffic_data1$Time_category <- cut(traffic_data1$Hour,
                                   breaks = time_intervals,
                                   labels = c("Night", "Morning", "Afternoon", "Evening"),
                                   include.lowest = TRUE)

traffic_data1 <- subset(traffic_data1, select = -c(Hour, Timenew))

summary(traffic_data1)


set.seed(123)
sample_size <- 53  

sampled_data <- traffic_data1 %>%
  group_by(Day, Time_category) %>%
  sample_n(sample_size, replace = FALSE)

summary(sampled_data)


ggplot(aes(x = NB_Speed), data = sampled_data) + geom_bar(stat = "count", fill = 'orange') + ggtitle("Northbound Speed")
ggplot(aes(x = SB_Speed), data = sampled_data) + geom_bar(stat = "count", fill = 'black') + ggtitle("Southbound Speed")

descriptive_stats <- sampled_data %>%
  group_by(Day, Time_category) %>%
  summarise(mean_speed_north = mean(NB_Speed, na.rm = TRUE),
            mean_speed_south = mean(SB_Speed, na.rm = TRUE),
            median_speed_north = median(NB_Speed, na.rm = TRUE),
            median_speed_south = median(SB_Speed, na.rm = TRUE),
            sd_speed_north = sd(NB_Speed, na.rm = TRUE),
            sd_speed_south = sd(SB_Speed, na.rm = TRUE))

print(descriptive_stats, n=30)

library(ggplot2)

plot_data <- tidyr::pivot_longer(descriptive_stats, 
                                 cols = c(mean_speed_north, mean_speed_south),
                                 names_to = "Direction",
                                 values_to = "Mean_Speed")

ggplot(plot_data, aes(x = Day, y = Mean_Speed, fill = Time_category)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Direction, scales = "free_y", ncol = 2) +
  labs(title = "Mean Traffic Speeds by Day and Time Category",
       x = "Day of Week",
       y = "Mean Speed",
       fill = "Time Category") +
  theme_minimal()


# comp mean speed across days and time 
hypothesis_test <- sampled_data %>%
  group_by(Day, Time_category) %>%
  summarise(p_value = t.test(NB_Speed, SB_Speed)$p.value)

print(hypothesis_test, n=28)

time_anova <- sampled_data %>%
  group_by(Time_category) %>%
  summarise(mean_speed_north = mean(NB_Speed, na.rm = TRUE),
            mean_speed_south = mean(SB_Speed, na.rm = TRUE)) %>%
  pivot_longer(cols = starts_with("mean_speed"),
               names_to = "Direction",
               values_to = "Mean_Speed") %>%
  ungroup() %>%
  ggplot(aes(x = Time_category, y = Mean_Speed, fill = Direction)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Mean Speeds During Different Driving Hours",
       x = "Time Category",
       y = "Mean Speed",
       fill = "Direction") +
  theme_minimal()

day_anova <- sampled_data %>%
  group_by(Day) %>%
  summarise(mean_speed_north = mean(NB_Speed, na.rm = TRUE),
            mean_speed_south = mean(SB_Speed, na.rm = TRUE)) %>%
  pivot_longer(cols = starts_with("mean_speed"),
               names_to = "Direction",
               values_to = "Mean_Speed") %>%
  ungroup() %>%
  ggplot(aes(x = Day, y = Mean_Speed, fill = Direction)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Mean Speeds Across Days",
       x = "Day",
       y = "Mean Speed",
       fill = "Direction") +
  theme_minimal()


print(day_anova)
print(time_anova)

any_na <- anyNA(sampled_data[, c("NB_Speed", "SB_Speed")])

if (any_na) {
  sampled_data1 <- na.omit(sampled_data[, c("NB_Speed", "SB_Speed")])
}
par(mfrow=c(1,1))
kmeans_cluster <- kmeans(sampled_data1, centers = 3, nstart = 10)
plot(sampled_data1$NB_Speed, sampled_data1$SB_Speed, 
     col = kmeans_cluster$cluster, 
     xlab = "Northbound Speed", ylab = "Southbound Speed",
     main = "K-means Clustering of Traffic Speeds")

-------
  # One-way ANOVA test to compare mean speeds across different days
  day_anova <- sampled_data %>%
  group_by(Day) %>%
  summarise(mean_speed_north = mean(NB_Speed, na.rm = TRUE),
            mean_speed_south = mean(SB_Speed, na.rm = TRUE)) %>%
  pivot_longer(cols = starts_with("mean_speed"),
               names_to = "Direction",
               values_to = "Mean_Speed") %>%
  ungroup() %>%
  ggplot(aes(x = Day, y = Mean_Speed, fill = Direction)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Mean Speeds Across Days",
       x = "Day",
       y = "Mean Speed",
       fill = "Direction") +
  theme_minimal()







