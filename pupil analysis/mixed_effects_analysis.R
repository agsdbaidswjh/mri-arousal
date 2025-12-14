library(lme4)
library(lmerTest) # p-values in lmer models
library(performance) # model comparison
library(ggplot2)
library(dplyr)
library(readr)
library(effectsize)

data <- read_csv("all_subjects_processed.csv")

# one sample t-test analysis on linear and quadratic beta coefficients 
betas_lin <- data %>%
  group_by(subject_id) %>%
  summarise(
    beta0_lin = dplyr::first(beta0_lin),
    beta1_lin = dplyr::first(beta1_lin),
    .groups = "drop"
  )

betas_quad <- data %>%
  group_by(subject_id) %>%
  summarise(
    beta0_quad = dplyr::first(beta0_quad),
    beta1_quad = dplyr::first(beta1_quad),
    beta2_quad = dplyr::first(beta2_quad),
    .groups = "drop"
  )

t1 <- t.test(betas_lin$beta1_lin, mu = 0)
cat("One-sample t-test for β1:\n")
print(t1)

t2 <- t.test(betas_quad$beta2_quad, mu = 0)
cat("\nOne-sample t-test for β2:\n")
print(t2)

cohens_d(betas_lin$beta1_lin, mu = 0, hedges.correction = FALSE, ci = 0.95)
cohens_d(betas_quad$beta2_quad, mu = 0, hedges.correction = FALSE, ci = 0.95)

# mixed effects models analysis
data <- data %>%
  mutate(subject_id = as.factor(subject_id))

linear_model <- lmer(RT_log ~ pupil_zscore + (1 | subject_id), 
                     data = data, REML = FALSE)
summary(linear_model)

linear_model_rs <- lmer(RT_log ~ pupil_zscore + (1 + pupil_zscore | subject_id),
                        data = data, REML = FALSE)
summary(linear_model_rs)

quad_model <- lmer(RT_log ~ pupil_zscore + pupil_zscore_squared + (1 | subject_id), 
                   data = data, REML = FALSE)
summary(quad_model)

quad_model_rs <- lmer(RT_log ~ pupil_zscore + pupil_zscore_squared + (1 + pupil_zscore | subject_id), 
                      data = data, REML = FALSE)
summary(quad_model_rs)

quad_model_rss <- lmer(RT_log ~ pupil_zscore + pupil_zscore_squared + (1 + pupil_zscore + pupil_zscore_squared | subject_id), 
                       data = data, REML = FALSE)
summary(quad_model_rss)

# linear intrc vs linear intrc + slope
AIC(linear_model, linear_model_rs); BIC(linear_model, linear_model_rs) # best model: linear_rs 

# quadratic intrc vs quadratic intrc + slope vs quadratic intrc + 2 slopes
AIC(quad_model, quad_model_rs, quad_model_rss); BIC(quad_model, quad_model_rs, quad_model_rss) # best model quad_rs 

anova(linear_model, linear_model_rs)
anova(quad_model, quad_model_rs)
anova(quad_model_rs, quad_model_rss)
anova(linear_model_rs, quad_model_rs)
anova(linear_model_rs, quad_model_rss)
