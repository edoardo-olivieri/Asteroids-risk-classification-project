
###PROGETTO DATA MINING: FRANCESCA VERNA, EDOARDO OLIVIERI

#VALUTAZIONE RISCHIO ASTEROIDI: ANALISI COMPARATIVA TRA METODI DI CLASSIFICAZIONE

nasa_nb <- read.csv("C:/Users/Hp/Downloads/nasa_nb.csv")

library(tidyverse)
library(skimr)
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(plotly)
library(caret)
library(e1071)
library(ggcorrplot)
library(GGally)
library(ISLR) 
library(class)
library(tree)
library(gridExtra)
library(car)
library(randomForest)
library(ROCR)
library(kableExtra)

#valutazione dati mancanti ed eventuali anomalie
skim(nasa_nb)

#prima selezione variabili
nasa_clean <- nasa_nb %>% select(-Neo.Reference.ID, -Name , -Close.Approach.Date, -Orbiting.Body, 
                                 -Orbit.ID, -Orbit.Determination.Date, -Equinox, -Asc.Node.Longitude, 
                                 -Est.Dia.in.Feet.max., -Est.Dia.in.Feet.min., -Est.Dia.in.M.max., 
                                 -Est.Dia.in.M.min., -Est.Dia.in.Miles.max., -Est.Dia.in.Miles.min., 
                                 -Epoch.Date.Close.Approach, -Relative.Velocity.km.per.sec, 
                                 -Miss.Dist..Astronomical., -Miss.Dist..lunar., -Miss.Dist..miles., 
                                 -Miles.per.hour, -Epoch.Osculation, -Perihelion.Time, -Orbital.Period) %>% 
  mutate(Hazardous=ifelse(Hazardous=="True", 1,0)) %>% mutate(Hazardous=as.factor(Hazardous))


## Splitting del dataset

set.seed(123)
nasa_idx = sample(nrow(nasa_clean), nrow(nasa_clean)*0.8) 
# Training Set
nasa_trn = nasa_clean[nasa_idx, ]
# Test Set
nasa_tst = nasa_clean[-nasa_idx,] 

set.seed(123)
subtrain_index <- sample(nrow(nasa_trn), nrow(nasa_trn)*0.65)
# Subtraining
sub_training <- nasa_trn[subtrain_index, ]
# Validation
validation <- nasa_trn[-subtrain_index, ]


# Validation
skim(validation)

# Test set
skim(nasa_tst)

# Subtraining
skim(sub_training)

## Bilanciamento
#grafico proporzioni
a <- round(prop.table(table(sub_training$Hazardous)),4)
b <- round(prop.table(table(validation$Hazardous)),4)
c <- round(prop.table(table(nasa_tst$Hazardous)),4)

dataset <- c(rep("Subtraining Set" , 2) , rep("Validation Set" , 2) , rep("Test Set" , 2))
esito <- rep(c("Non-Hazardous" , "Hazardous") , 3)
proporzione <- c(a,b,c)
data <- data.frame(dataset,esito,proporzione)

bar_plot <- ggplot(data, aes(fill=esito, y=proporzione, x=dataset)) + 
  geom_bar(position="fill", stat="identity") +
  scale_fill_manual(values = c("purple","green2")) +
  ggtitle("Proporzioni degli esiti")
ggplotly(bar_plot)

#upSample
set.seed(123)
sub_training <- upSample(sub_training[,-17], sub_training[,17], yname = "Hazardous")
table(sub_training$Hazardous)


## Boxplot del subtraining
melted_data <- reshape2::melt(sub_training, id.vars = "Hazardous") %>% rename(Variabile=variable) 

ggplot(melted_data, aes( x="", y = value, fill = Variabile)) +
  geom_boxplot( outlier.shape="x", outlier.size=3, show.legend = F) +
  facet_wrap(~ Variabile, nrow = 5, scales = "free") +
  labs(title = "Boxplots per 16 Variabili", x = "") +
  theme(axis.title.y=element_blank(),
        axis.title.x=element_blank())

#asimmetria
apply(sub_training[,-17], 2, skewness)

#trasformazione variabili

sub_trn_log <- sub_training %>% mutate(Est.Dia.in.KM.min.=log(Est.Dia.in.KM.min.),
                                       Est.Dia.in.KM.max.=log(Est.Dia.in.KM.max.),
                                       Minimum.Orbit.Intersection=log(Minimum.Orbit.Intersection),
                                       Inclination=log(Inclination))

#GRAFICO BOXPLOT POST TRASFORMAZIONI LOG

melted_data <- reshape2::melt(sub_trn_log, id.vars = "Hazardous") %>% rename(Variabile=variable) 

ggplot(melted_data, aes( x="", y = value, fill = Variabile)) +
  geom_boxplot( outlier.shape="x", outlier.size=3, show.legend = F) +
  facet_wrap(~ Variabile, nrow = 5, scales = "free") +
  labs(title = "Boxplots per 16 Variabili", x = "") +
  theme(axis.title.y=element_blank(),
        axis.title.x=element_blank())

#controllo asimmetria
apply(sub_trn_log[,-17], 2, skewness) #miglioramento, ma comunque si mantiene un po' di asimmetria in alcune variabili


## Correlazioni tra variabili

corr1 <- round(cor(sub_trn_log[,-c(17)]), 3)

corr_plot <- ggcorrplot(corr1, hc.order = TRUE, lab=T, lab_size = 4, type="lower", show.diag = T)
corr_plot


# Rimozione variabili altamente correlate
sub_trn_clean <- sub_trn_log %>% select(-Mean.Motion,-Semi.Major.Axis,-Aphelion.Dist,
                                        -Est.Dia.in.KM.min., -Est.Dia.in.KM.max.)

## Distribuzioni di densità

melted_data2 <- reshape2::melt(sub_trn_clean, id.vars = "Hazardous")

ggplot(melted_data2, aes(x = value)) +
  geom_density(aes(fill=factor(Hazardous)), alpha=0.5) +
  facet_wrap(~ variable, nrow = 4, scales = "free") +
  labs(title = "Distribuzioni di densità condizionatamente a Hazardous e Non-Hazardous", x = "Valore", y = "Densità") +
  scale_fill_manual(values = c("green2","purple"), name = "Esito", labels = c("Non-Hazardous",  "Hazardous")) +
  theme_minimal()

#selezione variabili che più discriminano tra le due classi
final_sub_trn <- sub_trn_clean %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                          Minimum.Orbit.Intersection,Hazardous)



# SUBTRAINING E VALIDATION
## Normalizzazione

#### Normalizzazione Subtraining (con trasformazioni effettuate) ####

minmax <- matrix(0, nrow=(dim(final_sub_trn)[2]-1), ncol=2) 
colnames(minmax) <- c("min", "max")

# calcolo min e max.
for (i in 1:(dim(final_sub_trn)[2]-1)){
  minmax[i, "min"] <- min(final_sub_trn[,i])
  minmax[i, "max"] <- max(final_sub_trn[,i])
}

# normalizzazione
for (i in 1:(dim(final_sub_trn)[2]-1)){
  final_sub_trn[, i] <- (final_sub_trn[, i] - minmax[i, "min"])/(minmax[i, "max"]- minmax[i, "min"])
}

summary(final_sub_trn)


#### Preparazione e normalizzazione Validation (con trasformazioni effettuate) ####

validation_log <- validation %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                        Minimum.Orbit.Intersection,Hazardous) %>%
  mutate(Minimum.Orbit.Intersection=log(Minimum.Orbit.Intersection))

# normalizzazione
for (i in 1:(dim(validation_log)[2]-1)){
  validation_log[, i] <- (validation_log[, i] - minmax[i, "min"])/(minmax[i, "max"]- minmax[i, "min"])
}

summary(validation_log)


## Analisi Discriminante

# Funzione per Q-Q plot per singola variabile
qq_plot <- function(data, var) {
  ggplot(data, aes(sample = !!sym(var), color = Hazardous)) + 
    geom_qq_line(color="black", linewidth=0.8) +
    geom_qq() + 
    facet_wrap(~Hazardous) +
    labs(title = paste("Q-Q Plot di", var), size = 2) +
    labs(x = "", y = "") +
    scale_color_manual(values = c("green2","purple"), name = "Esito", labels = c("Non-Hazardous",  "Hazardous"))
}

# Q-Q plot per le diverse variabili
plots_list <- lapply(names(final_sub_trn)[1:ncol(final_sub_trn)-1], function(var) qq_plot(final_sub_trn, var))

# Unione dei Q-Q plot
grid.arrange(grobs = plots_list, ncol = 2)



# Test Shapiro-Wilk diviso per classe
pvalue_shapiro <- final_sub_trn %>%
  group_by(Hazardous) %>%
  summarise(across(everything(), ~ shapiro.test(.x)$p.value, .names = "{.col}")) %>%
  pivot_longer(cols = -Hazardous, names_to = "Variable", values_to = "P_Value")

tab_shapiro <- pvalue_shapiro %>%
  pivot_wider(names_from = Hazardous, values_from = P_Value)
colnames(tab_shapiro) <- c("Variabile", "p-value Non-Hazardous", "p-value Hazardous")

# Tabella
shapiro_test <- tab_shapiro %>%
  kable("html", escape = FALSE, digits = 20) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "bordered"), full_width = FALSE)
shapiro_test

#non si prosegue con l'analisi discriminante


## Regressione Logistica

model_logit<-glm(Hazardous~., data=final_sub_trn, family=binomial)
summary(model_logit)

influencePlot(model_logit)

#rimozione punti influenti
final_sub_trn2<-final_sub_trn[-c(357, 668, 750, 1473), ]

model_logit2<-glm(Hazardous~., data=final_sub_trn2, family='binomial')
summary(model_logit2)#migliore

### Relazioni lineari
set.seed(123)
probabilities <- predict(model_logit2, type = "response")
predictors <- c("Absolute.Magnitude", "Orbit.Uncertainity", "Minimum.Orbit.Intersection")

# Costruzione delle log(p1/(1-p1))
supp <- final_sub_trn2[, -c(4)]
supp <- supp %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

# Costruzione dei grafici
ggplot(supp, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")



## K-Nearest Neighbors

final_sub_trn_KNN <- sub_training %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                             Minimum.Orbit.Intersection,Hazardous)
### Normalizzazione
minmax_KNN <- matrix(0, nrow=(dim(final_sub_trn_KNN)[2]-1), ncol=2) 
colnames(minmax_KNN) <- c("min", "max")

# calcolo min e max.
for (i in 1:(dim(final_sub_trn_KNN)[2]-1)){
  minmax_KNN[i, "min"] <- min(final_sub_trn_KNN[,i])
  minmax_KNN[i, "max"] <- max(final_sub_trn_KNN[,i])
}

# normalizzazione
for (i in 1:(dim(final_sub_trn_KNN)[2]-1)){
  final_sub_trn_KNN[, i] <- (final_sub_trn_KNN[, i] - minmax_KNN[i, "min"])/(minmax_KNN[i, "max"]- minmax_KNN[i, "min"])
}

summary(final_sub_trn_KNN)


#### Preparazione (e norm) Validation KNN ####

validation_KNN <- validation %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                        Minimum.Orbit.Intersection,Hazardous)

# normalizzazione
for (i in 1:(dim(validation_KNN)[2]-1)){
  validation_KNN[, i] <- (validation_KNN[, i] - minmax_KNN[i, "min"])/(minmax_KNN[i, "max"]- minmax_KNN[i, "min"])
}
summary(validation_KNN)


### Ricerca del k ottimale

err_rate = function(actual, predicted) { mean(actual != predicted) }

set.seed(123)
k_to_try = 1:100
err_k = rep(x = 0, times = length(k_to_try))
for (i in seq_along(k_to_try)) {
  pred = knn(train = final_sub_trn_KNN[,-4],
             test = validation_KNN[,-4], cl = final_sub_trn_KNN[,4],
             k = k_to_try[i])
  err_k[i] = err_rate(validation_KNN[,4], pred) }

plot(err_k, type = "b", col = "cornflowerblue", cex = 1, pch = 20,
     xlab = "K", ylab = "Error rate", 
     main = "Error rate vs K")
#k ottimale:3


# PREVISIONE SU VALIDATION

## Regressione Logistica

predAIC<-ifelse(predict(model_logit2, validation_log, type='response')>0.5, 1, 0)
(conf_matrixreg<-confusionMatrix(as.factor(predAIC), validation_log[, 4], positive='1'))

## KNN

set.seed(123)
pred_knn <-  knn(train = final_sub_trn_KNN[,-4], test = validation_KNN[,-4], cl = final_sub_trn_KNN[,4],k = 3, prob = TRUE)
(conf_matrix<-confusionMatrix(as.factor(pred_knn),as.factor(validation_KNN$Hazardous), positive="1"))

#False Negative Rate KNN
conf_matrix$table[1,2]/(conf_matrix$table[1,1]+conf_matrix$table[1,2])

## F1 score
# F1 score regr logistica
F1_regr_log <-conf_matrixreg$byClass[7]
F1_regr_log

# F1 score KNN
F1_KNN <- conf_matrix$byClass[7]
F1_KNN


## Curve di ROC e auc

#auc regressione logistica
pred_roclogit <- prediction(predAIC, validation$Hazardous)
perf_logit <- performance(pred_roclogit,"tpr","fpr")
auc_logit <- performance(pred_roclogit, measure = "auc")@y.values
cat("auc Regressione Logistica", ": ", auc_logit[[1]])

#auc KNN
prob_knn_val <- attr(pred_knn, "prob")
prob_knn_val <- 2*ifelse(pred == "0", 1-prob_knn_val, prob_knn_val) -1
pred_rocknn_val <- prediction(prob_knn_val, validation_KNN[, 4])
pred_knn_val <- performance(pred_rocknn_val, "tpr", "fpr")
auc_knn_val <- performance(pred_rocknn_val, measure = "auc")@y.values
cat("auc KNN", ": ", auc_knn_val[[1]])

#grafici curve di ROC
par(mfrow=c(1,2))
plot(perf_logit, colorize=T, lwd=3, main="Regressione Logistica su Validation", cex.main = 1) ; abline(a=0, b=1, lty=2, col="gray")
plot(pred_knn_val, colorize=T, lwd=3, main="KNN su Validation", cex.main = 1); abline(a=0, b=1, lty=2, col="gray")

par(mfrow=c(1,1))
#KNN è il migliore: si decide di non continuare con la regressione logistica


# TRAINING E TEST
## K-Nearest Neighbors

training_KNN <- nasa_trn %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                    Minimum.Orbit.Intersection,Hazardous)
### Bilanciamento
set.seed(123)
training_KNN <- upSample(training_KNN[,-4], training_KNN[,4], yname="Hazardous")
table(training_KNN$Hazardous)

### Normalizzazione

 
minmax_TRN_KNN <- matrix(0, nrow=(dim(training_KNN)[2]-1), ncol=2) 
colnames(minmax_TRN_KNN) <- c("min", "max")

# calcolo min e max.
for (i in 1:(dim(training_KNN)[2]-1)){
  minmax_TRN_KNN[i, "min"] <- min(training_KNN[,i])
  minmax_TRN_KNN[i, "max"] <- max(training_KNN[,i])
}

# normalizzazione
for (i in 1:(dim(training_KNN)[2]-1)){
  training_KNN[, i] <- (training_KNN[, i] - minmax_TRN_KNN[i, "min"])/(minmax_TRN_KNN[i, "max"]- minmax_TRN_KNN[i, "min"])
}

summary(training_KNN)

#preparazione test_KNN

test_KNN <- nasa_tst %>% select(Absolute.Magnitude, Orbit.Uncertainity,
                                Minimum.Orbit.Intersection,Hazardous)

# summary(test_KNN)

# normalizzazione
for (i in 1:(dim(test_KNN)[2]-1)){
  test_KNN[, i] <- (test_KNN[, i] -minmax_TRN_KNN[i, "min"])/(minmax_TRN_KNN[i, "max"]- minmax_TRN_KNN[i, "min"])
}


summary(test_KNN)


### Classification Boundaries

#knn su training_KNN con k=3
set.seed(123)
model <- knn(training_KNN[, -4], test_KNN[, -4], cl=training_KNN[, 4], k=3, prob=TRUE)


x1_range <- seq(min(training_KNN$Absolute.Magnitude)-0.2, max(training_KNN$Absolute.Magnitude)+0.2, length.out = 20)
x2_range <- seq(min(training_KNN$Orbit.Uncertainity)-0.2, max(training_KNN$Orbit.Uncertainity)+0.2, length.out = 20)
x3_range <- seq(min(training_KNN$Minimum.Orbit.Intersection)-0.2, max(training_KNN$Minimum.Orbit.Intersection)+0.2, length.out = 20)
grid <- expand.grid(Absolute.Magnitude = x1_range, Orbit.Uncertainity = x2_range, Minimum.Orbit.Intersection = x3_range)
grid$pred <- knn(training_KNN[,-4], grid, training_KNN[, 4], k = 3)

# Grafico interattivo classification boundaries KNN
plot_ly(data = grid, y = ~Absolute.Magnitude, x = ~Orbit.Uncertainity, z = ~Minimum.Orbit.Intersection, color = ~as.factor(grid$pred), colors = c("green2", "purple"), type = "scatter3d", mode = "markers", marker = list(size = 1.6, opacity = 1)) %>%
  layout(scene = list(yaxis = list(title = "Absolute.Magnitude"),
                      xaxis = list(title = "Orbit.Uncertainity"),
                      zaxis = list(title = "Minimum.Orbit.Intersection"),
                      camera = list(eye = list(x = -1.8, y = -2, z = 1.4))))


# Grafico interattivo con osservazioni del training sovraimposte
plot_ly(data = grid, y = ~Absolute.Magnitude, x = ~Orbit.Uncertainity, z = ~Minimum.Orbit.Intersection, color = ~as.factor(grid$pred), colors = c("green2", "purple"), type = "scatter3d", mode = "markers", marker = list(size = 1.2, opacity = 0.7)) %>% add_trace(data = training_KNN[,-4], y = ~Absolute.Magnitude, x = ~Orbit.Uncertainity, z = ~Minimum.Orbit.Intersection, color = ~as.factor(training_KNN$Hazardous), colors = c("green2", "purple"), type = "scatter3d", mode = "markers", marker = list(size = 3, opacity = 0.8)) %>%
  layout(scene = list(yaxis = list(title = "Absolute.Magnitude"),
                      xaxis = list(title = "Orbit.Uncertainity"),
                      zaxis = list(title = "Minimum.Orbit.Intersection"),
                      camera = list(eye = list(x = -1.8, y = -2, z = 1.4))))



## Alberi Decisionali

### Bilanciamento

set.seed (123)
nasa_trn <- upSample(nasa_trn[,-17], nasa_trn[,17], yname="Hazardous")
table(nasa_trn$Hazardous)

### Costruzione dell'albero

set.seed (123)

tree.nasa.trn <- tree(Hazardous ~. , nasa_trn)

plot(tree.nasa.trn) ; text(tree.nasa.trn, pretty = 0) ; title(main = "Albero di classificazione non potato")


tree.pred1 <- predict(tree.nasa.trn , nasa_tst , type = "class")
confusionMatrix(tree.pred1, nasa_tst$Hazardous, positive="1")

### CV e pruning

set.seed (123)
cv.nasa <- cv.tree(tree.nasa.trn , FUN = prune.misclass)
#SCELGO LA SIZE MIGLIORE
plot(cv.nasa$size , cv.nasa$dev, type = "b")

prune.nasa <- prune.misclass(tree.nasa.trn , best = 3)
#GRAFICO POTATO
plot(prune.nasa) ; text(prune.nasa , pretty = 0) ; title(main = "Albero di classificazione potato")

## Random Forest

### Primo step

set.seed(123)
rf_trn_1 <- randomForest(Hazardous ~ ., nasa_trn, importance = T, proximity = T, ntree = 200)
rf_trn_1


### Selezione delle variabili

varImpPlot(rf_trn_1, 
           main = "Importanza delle variabili per la distinzione tra Hazardous e Non-Hazardous")

nasa_trn2 <- nasa_trn %>% select(Minimum.Orbit.Intersection, Orbit.Uncertainity, Est.Dia.in.KM.min., Est.Dia.in.KM.max., Perihelion.Distance, Absolute.Magnitude, Hazardous)

### Secondo step

set.seed(123)
rf_trn_2 <- randomForest(Hazardous ~ ., nasa_trn2, importance = T, proximity = T, ntree = 200)
rf_trn_2

### Numero di alberi ottimale

plot(rf_trn_2) #ntree=50

### Terzo step

set.seed(123)
rf_trn_3 <- randomForest(Hazardous ~ ., nasa_trn2, importance = T, proximity = T, ntree = 50)

# PREVISIONE SU TEST

conf_KNN <- confusionMatrix(data=model,reference=factor(test_KNN[, 4]), positive="1")
F1_KNN <- conf_KNN$byClass[[7]]
sens_KNN <- conf_KNN$byClass[[1]]
spec_KNN <- conf_KNN$byClass[[2]]
fnr_KNN <- 1-conf_KNN$byClass[[1]]
conf_KNN

## Alberi Decisionali

set.seed(123)
tree.pred2 <- predict(prune.nasa , nasa_tst , type = "class")
conf_DT <- confusionMatrix(tree.pred2, nasa_tst$Hazardous, positive="1")
F1_DT <- conf_DT$byClass[[7]]
sens_DT <- conf_DT$byClass[[1]]
spec_DT <- conf_DT$byClass[[2]]
fnr_DT <- 1-conf_DT$byClass[[1]]
conf_DT

## RandomForest

set.seed(123)
rf_pred <- predict(rf_trn_3, nasa_tst) %>%
  as.data.frame() %>%
  mutate(Hazardous = as.factor(`.`)) %>%
  select(Hazardous)

conf_RF <- confusionMatrix(rf_pred$Hazardous, nasa_tst$Hazardous, positive="1")
F1_RF <- conf_RF$byClass[[7]]
sens_RF <- conf_RF$byClass[[1]]
spec_RF <- conf_RF$byClass[[2]]
fnr_RF <- 1-conf_RF$byClass[[1]]
conf_RF

## Tabella Riassuntiva

tab <- data.frame(
  Sensitivity = c(sens_KNN, sens_DT, sens_RF),
  Specificity = c(spec_KNN, spec_DT,spec_RF),
  False_Negative_Rate = c(fnr_KNN, fnr_DT, fnr_RF),
  F1_score = c(F1_KNN, F1_DT, F1_RF)
)

rownames(tab) <- c("KNN", "Alberi Decisionali", "RandomForest")

tab[1,1] <- cell_spec(tab[1, 1], color = "green3")
tab[1,2] <- cell_spec(tab[1, 2], color = "red3")
tab[1,3] <- cell_spec(tab[1, 3], color = "green3")
tab[1,4] <- cell_spec(tab[1, 4], color = "red3")

tab[2,1] <- cell_spec(tab[2, 1], color = "red3")
tab[2,2] <- cell_spec(tab[2, 2], color = "orange2")
tab[2,3] <- cell_spec(tab[2, 3], color = "red3")
tab[2,4] <- cell_spec(tab[2, 4], color = "orange2")

tab[3,1] <- cell_spec(tab[3, 1], color = "green3")
tab[3,2] <- cell_spec(tab[3, 2], color = "green3")
tab[3,3] <- cell_spec(tab[3, 3], color = "green3")
tab[3,4] <- cell_spec(tab[3, 4], color = "green3")


tabella <- tab %>%
  kable("html", escape=FALSE) %>%
  kable_styling(bootstrap_options = c("striped","hover","bordered"))
tabella
#Random Forest migliore, seguito da KNN e Alberi Decisionali

## Visualizzazione del RF

rf_class <- data.frame(actual = nasa_tst$Hazardous,
                       predicted = rf_pred$Hazardous) %>%
  mutate(Status = ifelse(actual == predicted, "CORRETTI" , "SBAGLIATI"))


ggplot(data = rf_class, mapping = aes(x = predicted, y = actual, 
                                      color = Status, shape = Status)) +
  scale_color_manual(values = c("green2","red2")) +
  geom_jitter(size = 2, alpha = 0.8) +
  labs(x = "Esito previsto",
       y = "Esito reale", title= "Previsioni con modello RandomForest (tree = 50)") +
  theme_bw()
#3 osservazioni misclassificate in rosso

## Curve di ROC

prob_knn <- attr(model, "prob")
prob_knn <- 2*ifelse(model == "0", 1-prob_knn, prob_knn) -1
pred_rocknn <- prediction(prob_knn, test_KNN[, 4])
pred_knn <- performance(pred_rocknn, "tpr", "fpr")
auc_knn <- performance(pred_rocknn, measure = "auc")@y.values

tree.preds <- predict(prune.nasa, nasa_tst, type="vector")[,"1"]
pred_roctree <- prediction(tree.preds, nasa_tst$Hazardous)
pred_tree <- performance(pred_roctree, "tpr", "fpr")
auc_tree <- performance(pred_roctree, measure = "auc")@y.values

pred_rocRF <- predict(rf_trn_3, newdata=nasa_tst, type="prob")
pred_rocRF <- pred_rocRF[,"1"]
pred_rocRF <- prediction(pred_rocRF, nasa_tst$Hazardous)

pred_RF <- performance(pred_rocRF, "tpr", "fpr")
auc_RF <- performance(pred_rocRF, measure = "auc")@y.values

par(mfrow=c(1,3))
plot(pred_knn, colorize=T, lwd=3) ; abline(a=0, b=1, lty=2, col="gray") ; title(main = "Curva di ROC per KNN", sub = paste("AUC pari a:", round(auc_knn[[1]],3)), cex.sub = 1.4) 
plot(pred_tree, colorize=T, lwd=3, main="Curva di ROC per Alberi Decisionali"); abline(a=0, b=1, lty=2, col="gray"); title(sub = paste("AUC pari a:", round(auc_tree[[1]],3)), cex.sub = 1.4) 
plot(pred_RF, colorize=T, lwd=3, main="Curva di ROC per RF"); abline(a=0, b=1, lty=2, col="gray"); title(sub = paste("AUC pari a:", round(auc_RF[[1]],5)), cex.sub = 1.4)

#migliore Random Forest, seguito da KNN e Alberi Decisionali