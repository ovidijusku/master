install.packages("fda")
install.packages("tidyr")
install.packages("lubridate")
install.packages("tidyverse")
install.packages("fda.usc")
install.packages("janitor")
install.packages("funFEM")

library(fda)
library(dplyr)
library(tidyr)
library(lubridate)
library(tidyverse)
library(fds)
library(fda.usc)
library(janitor)
library(funFEM)

df <- read.csv("temp_only_20_countries.csv")

df <- df %>% 
  mutate (date = as.Date(dt), month = month(dt), year = year(dt))

df_pivot <- df %>%
  select(dt, year, month, country, avg_temp) %>%
  group_by(dt, country) %>%
  pivot_wider(names_from = country, values_from = avg_temp) %>%
  ungroup()

df_y30 <- df_pivot %>% select(-dt, -year, -month)

y30_Rng  = c(1,360)
y30_per  = seq(1, 360)

nbasis = 80;
y30_basis10 = create.fourier.basis(y30_Rng, nbasis)

fdnames      = vector('list', 3)
fdnames[[1]] = "Period"
fdnames[[2]] = "Country"
fdnames[[3]] = "Average Temperature"

basismat   = eval.basis(y30_per, y30_basis10)
y30_mat  = apply(df_y30, 2, as.numeric)
y30_coef = lsfit(basismat, y30_mat, intercept=FALSE)$coef

y30_fd <- fd(y30_coef, y30_basis10, fdnames)

y30_List = smooth.basis(y30_per, y30_mat, y30_basis10)
y30_fd   = y30_List$fd
plot(y30_fd)

# PCA
#nharm = 2
#pcalist = pca.fd(y30_fd, nharm, centerfns = FALSE)
#plot(pcalist$harmonics, xlab="months", ylab="temperature", main="FPCA for Temperatures Across 20 Countries")
#pcalist$varprop

#legend("bottomleft", legend = paste("FPC", 1:2), col = 1:2, pch = 1, cex = 0.8)


# HYPOTHESIS TESTING
# H0: m(temp)_first_15_years = m(temp)_last_15_years
# H1: m(temp)_first_15_years =/= m(temp)_last_15_years
# Assumption: independent samples
source("trace.R")
source("L2stattwosample.R")
source("Ztwosample.R")

y15_Rng  = c(1,180)
y15_per  = seq(1, 180)
nbasis = 29

# first 15 years
y15_basis = create.fourier.basis(y15_Rng, nbasis)
basismat   = eval.basis(y15_per, y15_basis)
y15_mat  = apply(df_y30[1:180,], 2, as.numeric)
y15_coef = lsfit(basismat, y15_mat, intercept=FALSE)$coef
y15_fd_before <- fd(y15_coef, y15_basis, fdnames)
y15_List = smooth.basis(y15_per, y15_mat, y15_basis)
y15_fd_before_final   = y15_List$fd
plot(y15_fd_before)

# last 15 years
y15_basis = create.fourier.basis(y15_Rng, nbasis)
basismat   = eval.basis(y15_per, y15_basis)
relevant_df <- df_y30[181:360,]
rownames(relevant_df) <- NULL
y15_mat  = apply(relevant_df, 2, as.numeric)
y15_coef = lsfit(basismat, y15_mat, intercept=FALSE)$coef
y15_fd_after <- fd(y15_coef, y15_basis, fdnames)
y15_List = smooth.basis(y15_per, y15_mat, y15_basis)
y15_fd_after_final   = y15_List$fd

# Plotting mean functions
mean_y15_fd_before <- mean.fd(y15_fd_before_final)
mean_y15_fd_after <- mean.fd(y15_fd_after_final)
plot(mean_y15_fd_before, col="blue", lwd=2, main="Mean Functions (First 15 Years)")
lines(mean_y15_fd_after, col="red", lwd=2)
legend("bottomright", legend=c("First 15 Years", "Last 15 Years"), col=c("blue", "red"), lwd=2)

# L2 test
stat <- L2.stat.twosample(x=y15_fd_before, y=y15_fd_after, t.seq = y15_per, method=1)
stat
stat <- L2.stat.twosample(x=y15_fd_before, y=y15_fd_after, t.seq = y15_per, method=2, replications=500)
stat$pvalue
stat$statistics

# Two sample Z test
stat <- Ztwosample(x=y15_fd_before, y=y15_fd_after, t.seq = y15_per)
stat

# H0: mean(temp)_Europe = m(temp)_Africa
# H1: mean(temp)_Europe =/= mean(temp)_Africa
# Assumption: representative sample

categorize_region <- function(country) {
  region_mapping <- list(
    Europe = c("Monaco", "Germany", "Belgium", "Sweden", "Liechtenstein"),
    Asia = c("Angola", "Burundi", "Chad", "Rwanda", "Senegal", "Sudan", "Tanzania"),
    Africa = c("Kazakhstan", "Malaysia", "Taiwan")
  )
  for (region in names(region_mapping)) {
    if (country %in% region_mapping[[region]]) {
      return(region)
    }
  }
  return("Other")
}

df$region <- sapply(df$country, categorize_region)

df_pivot <- df %>%
  group_by(dt, region) %>%
  summarize(avg_temp = mean(avg_temp, na.rm = TRUE)) %>%
  pivot_wider(names_from = region, values_from = avg_temp) %>%
  ungroup()

df_y30 <- df_pivot %>% select(-dt)

y30_Rng  = c(1,360)
y30_per  = seq(1, 360)

nbasis = 29;
y30_basis10 = create.fourier.basis(y30_Rng, nbasis)

fdnames      = vector('list', 3)
fdnames[[1]] = "Period"
fdnames[[2]] = "Region"
fdnames[[3]] = "Average Temperature"

basismat   = eval.basis(y30_per, y30_basis10)
y30_mat  = apply(df_y30, 2, as.numeric)
y30_coef = lsfit(basismat, y30_mat, intercept=FALSE)$coef

y30_fd <- fd(y30_coef, y30_basis10, fdnames)

y30_List = smooth.basis(y30_per, y30_mat, y30_basis10)
y30_fd   = y30_List$fd
plot(y30_fd)
legend("topright", legend = y30_fd$fdnames[[2]], col = 1:length(y30_fd$fdnames[[2]]), lty = 1)

y30_fd_deriv <- deriv.fd(y30_fd)
plot(y30_fd_deriv)
legend("topright", legend = y30_fd_deriv$fdnames[[2]], col = 1:length(y30_fd_deriv$fdnames[[2]]), lty = 1)


# H0: derivative(temp)_Europe = derivative(temp)_Asia
# H1: derivative(temp)_Europe =/= derivative(temp)_Asia
# Assumption: representative sample
stat <- L2.stat.twosample(x=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Europe"], y=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Africa"],
                          t.seq = y30_per, method=2, replications=50)
stat

# H0: derivative(temp)_Europe = derivative(temp)_Asia
# H1: derivative(temp)_Europe =/= derivative(temp)_Asia
# Assumption: representative sample
stat <- L2.stat.twosample(x=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Europe"], y=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Asia"],
                          t.seq = y30_per, method=2, replications=50)
stat

# H0: derivative(temp)_Africa = derivative(temp)_Asia
# H1: derivative(temp)_Africa =/= derivative(temp)_Asia
# Assumption: representative sample
stat <- L2.stat.twosample(x=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Africa"], y=y30_fd_deriv[y30_fd_deriv$fdnames[[2]] == "Asia"],
                          t.seq = y30_per, method=2, replications=50)
stat

### CLUSTERING
df_pivot <- df %>%
  select(dt, year, month, country, avg_temp) %>%
  group_by(dt, country) %>%
  pivot_wider(names_from = country, values_from = avg_temp) %>%
  ungroup()

df_y30 <- df_pivot %>% select(-dt, -year, -month)
nbasis = 29
y30_basis = create.fourier.basis(y30_Rng, nbasis)
fdnames      = vector('list', 2)
fdnames[[1]] = "Period"
fdnames[[2]] = "Country"
fdnames[[3]] = "Average Temperature"
basismat   = eval.basis(y30_per, y30_basis)
y30_mat  = apply(df_y30, 2, as.numeric)
y30_coef = lsfit(basismat, y30_mat, intercept=FALSE)$coef
y30_fd <- fd(y30_coef, y30_basis, fdnames)

# 2 clusters (puts Northern hemosphere countries which are >30 deg lattitudes into cluster 1 and other into cluster 2)
res_w <- funFEM(y30_fd, K=2, model='all')
dt_w <- data.frame(country =  colnames(df_y30),
                   cluster = res_w$cls)
dt_w$data <- t(summary(df_y30))
par(mfrow=c(1,2))
plot(y30_fd, col=res_w$cls, lwd=2, lty=1)
fdmeans_w <- y30_fd
fdmeans_w$coefs <- t(res_w$prms$my)
plot(fdmeans_w, col=1:max(res_w$cls), lwd=2)

# 3 clusters (first cluster remains the same, splits second cluster into two: where 3rd holds higher temperatures and 2nd lower)
res_w <- funFEM(y30_fd, K=3, model='all')
dt_w <- data.frame(country =  colnames(df_y30),
                   cluster = res_w$cls)
dt_w$data <- t(summary(df_y30))
plot(y30_fd, col=res_w$cls, lwd=2, lty=1)
fdmeans_w <- y30_fd
fdmeans_w$coefs <- t(res_w$prms$my)
plot(fdmeans_w, col=1:max(res_w$cls), lwd=2)

par(mfrow=c(1,2))
dt_w