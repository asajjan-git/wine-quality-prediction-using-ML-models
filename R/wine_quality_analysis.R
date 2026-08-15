# =============================================================
# Wine Quality Prediction Using Machine Learning
# -------------------------------------------------------------
# Predicts the sensory quality score of white "Vinho Verde" wine
# from 11 physicochemical measurements. Compares Multiple Linear
# Regression, Random Forest, and XGBoost.
#
# Dataset: UCI Wine Quality (white wine subset), Cortez et al. (2009)
#
# Usage:
#   Rscript wine_quality_analysis.R
# =============================================================

required_packages <- c("randomForest", "xgboost", "ggplot2", "corrplot", "caret")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages, repos = "https://cloud.r-project.org")

library(randomForest)
library(xgboost)
library(ggplot2)
library(corrplot)
library(caret)

set.seed(42)

# --- Paths ---
data_dir <- file.path("..", "data")
fig_dir <- file.path("..", "figures")
results_dir <- file.path("..", "results")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

data_path <- file.path(data_dir, "winequality-white.csv")
data_url <- paste0(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/",
  "wine-quality/winequality-white.csv"
)

# --- 1. Load data (download if needed) ---
if (!file.exists(data_path)) {
  message("Dataset not found locally, downloading from UCI...")
  download.file(data_url, destfile = data_path, mode = "wb")
}
wine <- read.csv(data_path, sep = ";")

cat("\n=== Dataset dimensions ===\n")
print(dim(wine))

cat("\n=== Missing values ===\n")
print(sum(is.na(wine)))

cat("\n=== Summary statistics ===\n")
print(summary(wine))

# --- 2. Correlation heatmap ---
png(file.path(fig_dir, "correlation_heatmap.png"), width = 900, height = 800)
corr_matrix <- cor(wine)
corrplot(corr_matrix, method = "color", type = "upper", tl.col = "black",
         tl.srt = 45, title = "Correlation Heatmap of Features", mar = c(0, 0, 2, 0))
dev.off()
cat("\nSaved correlation heatmap ->", file.path(fig_dir, "correlation_heatmap.png"), "\n")

cat("\nCorrelation with quality (sorted):\n")
print(sort(corr_matrix[, "quality"], decreasing = TRUE))

# --- 3. Train/test split (70/30) ---
train_index <- createDataPartition(wine$quality, p = 0.70, list = FALSE)
train <- wine[train_index, ]
test <- wine[-train_index, ]

# Z-score scaling for the linear model (fit on train, applied to test)
preproc <- preProcess(train[, -ncol(train)], method = c("center", "scale"))
train_scaled <- predict(preproc, train)
test_scaled <- predict(preproc, test)

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae <- function(actual, pred) mean(abs(actual - pred))
r_squared <- function(actual, pred) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

results <- data.frame(Model = character(), RMSE = double(), R_squared = double(), MAE = double())

# --- 4. Multiple Linear Regression ---
lm_model <- lm(quality ~ ., data = train_scaled)
cat("\n=== Multiple Linear Regression summary ===\n")
print(summary(lm_model))

pred_lm <- predict(lm_model, newdata = test_scaled)
results <- rbind(results, data.frame(
  Model = "Multiple Linear Regression",
  RMSE = rmse(test$quality, pred_lm),
  R_squared = r_squared(test$quality, pred_lm),
  MAE = mae(test$quality, pred_lm)
))

# --- 5. Random Forest ---
rf_model <- randomForest(quality ~ ., data = train, ntree = 500, importance = TRUE)
print(rf_model)

png(file.path(fig_dir, "feature_importance_random_forest.png"), width = 900, height = 600)
varImpPlot(rf_model, main = "Random Forest — Feature Importance")
dev.off()
cat("\nSaved RF feature importance ->", file.path(fig_dir, "feature_importance_random_forest.png"), "\n")

pred_rf <- predict(rf_model, newdata = test)
results <- rbind(results, data.frame(
  Model = "Random Forest",
  RMSE = rmse(test$quality, pred_rf),
  R_squared = r_squared(test$quality, pred_rf),
  MAE = mae(test$quality, pred_rf)
))

# --- 6. XGBoost ---
train_matrix <- xgb.DMatrix(data = as.matrix(train[, -ncol(train)]), label = train$quality)
test_matrix <- xgb.DMatrix(data = as.matrix(test[, -ncol(test)]), label = test$quality)

xgb_model <- xgb.train(
  params = list(objective = "reg:squarederror", max_depth = 6, eta = 0.1),
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
  early_stopping_rounds = 10,
  verbose = 0
)

importance_matrix <- xgb.importance(model = xgb_model)
png(file.path(fig_dir, "feature_importance_xgboost.png"), width = 900, height = 600)
xgb.plot.importance(importance_matrix, main = "XGBoost — Feature Importance")
dev.off()
cat("\nSaved XGBoost feature importance ->", file.path(fig_dir, "feature_importance_xgboost.png"), "\n")

pred_xgb <- predict(xgb_model, test_matrix)
results <- rbind(results, data.frame(
  Model = "XGBoost",
  RMSE = rmse(test$quality, pred_xgb),
  R_squared = r_squared(test$quality, pred_xgb),
  MAE = mae(test$quality, pred_xgb)
))

# --- 7. Compare & save ---
results <- results[order(results$RMSE), ]
cat("\n=== Final Comparison ===\n")
print(results)

write.csv(results, file.path(results_dir, "metrics_summary.csv"), row.names = FALSE)
cat("\nSaved metrics ->", file.path(results_dir, "metrics_summary.csv"), "\n")

results_long <- reshape(
  results,
  varying = c("RMSE", "R_squared", "MAE"),
  v.names = "Value",
  timevar = "Metric",
  times = c("RMSE", "R_squared", "MAE"),
  direction = "long"
)

p <- ggplot(results_long, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Metric, scales = "free_y") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "none") +
  labs(title = "Model Performance Comparison", x = NULL, y = NULL)

ggsave(file.path(fig_dir, "model_comparison.png"), plot = p, width = 10, height = 5, dpi = 150)
cat("\nSaved model comparison chart ->", file.path(fig_dir, "model_comparison.png"), "\n")
