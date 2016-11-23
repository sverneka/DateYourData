### R script for generating few features
###########################################################
#####################################calculating Experience
student <- read.csv("Student.csv")
studentExp <- student[,c(1,18,19)]
studentExp$End.Date <- as.Date(studentExp$End.Date, format = c("%d-%m-%Y"))
studentExp$Start.Date <- as.Date(studentExp$Start.Date, format = c("%d-%m-%Y"))
studentExp$exp <- as.double(studentExp$End.Date - studentExp$Start.Date)
studentExp$exp[is.na(studentExp$exp)] <- 0
aggregate(exp~Student_ID, data=student, FUN=sum)
final <- aggregate(exp~Student_ID, data=studentExp, FUN=sum)
write.csv(final,"studentExp.csv")

#############################################################
############################ calculating stipend within range
require(plyr)
internship <- read.csv("Internship.csv")
internship <- internship[,c(1,9,10)]
internship$Stipend1 <- as.character(internship$Stipend1)
internship$Stipend1[internship$Stipend1=="NULL"] = "0"
internship$Stipend1 <- as.numeric(internship$Stipend1)
internship$Stipend2 <- as.character(internship$Stipend2)
for ( i in c(1:nrow(internship))) {
  print(i)
  if(internship[i,3]=="NULL") {
    internship[i,3]=internship[i,2]
  }
}
internship$Stipend2 <- as.numeric(internship$Stipend2)

train <- read.csv("Train.csv")
train <- train[,c(1,2,4)]
train$Expected_Stipend <- 
  mapvalues(train$Expected_Stipend, from =
              levels(train$Expected_Stipend), to = c(10,2,5,0))
train$Expected_Stipend <- as.numeric(as.character(train$Expected_Stipend))
train$expInRange <- 0
train$expDiff <- 0
for (i in c(1:nrow(train))) {
  print(i)
  temp <- internship[(internship$Internship_ID==train[i,1]),3]
  train[i,4]  <- (((train[i,3]) * 1000) <= temp)
  train[i,5] <- (temp - ((train[i,3]) * 1000))
}
train <- train[,c(1,2,4,5)]
write.csv(train,"trainExpStipend.csv")

### same thing for test file
test <- read.csv("test.csv")
test <- test[,c(1,2,4)]
test$Expected_Stipend <- 
  mapvalues(test$Expected_Stipend, from =
              levels(test$Expected_Stipend), to = c(10,2,5,0))
test$Expected_Stipend <- as.numeric(as.character(test$Expected_Stipend))
test$expInRange <- 0
test$expDiff <- 0
for (i in c(1:nrow(test))) {
  print(i)
  temp <- internship[(internship$Internship_ID==test[i,1]),3]
  test[i,4]  <- (((test[i,3]) * 1000) <= temp)
  test[i,5] <- (temp - ((test[i,3]) * 1000))
}
test <- test[,c(1,2,4,5)]
write.csv(test,"testExpStipend.csv")

####################################################################
##################################### Calculating location difference
internship <- read.csv("Internship.csv")
internship <- internship[,c(1,5)]
internship$Internship_Location <- as.character(internship$Internship_Location)

train <- read.csv("train.csv")
train <- train[,c(1,2,6)]
train$Preferred_location <- as.character(train$Preferred_location)
train$matchLocation <- 0
for (i in c(1:nrow(train))) {
  print(i)
  temp <- internship[(internship$Internship_ID==train[i,1]),2]
  train[i,4] <- (train[i,3]==temp)  
}
write.csv(train,"trainLocationMatch.csv")

test <- read.csv("test.csv")
test <- test[,c(1,2,6)]
test$Preferred_location <- as.character(test$Preferred_location)
test$matchLocation <- 0
for (i in c(1:nrow(test))) {
  print(i)
  temp <- internship[(internship$Internship_ID==test[i,1]),2]
  test[i,4] <- (test[i,3]==temp)  
}
write.csv(test,"testLocationMatch.csv")

#############################################################
#############################################################