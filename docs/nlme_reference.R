# Authored-but-not-executed R reference.
# This project was built and validated in Python (see src/) because R is
# not available in this execution environment. This file documents how the
# same population PK model would be fit in R using the standard nlmixr2
# population-PK stack, for reference/portability.

library(nlmixr2)

pk_data <- read.csv("simulated_pk_data.csv")  # id, time, dose, dv (concentration), weight, age

one_cmt_oral <- function() {
  ini({
    tcl <- log(5)     # typical clearance
    tv  <- log(50)    # typical volume
    tka <- log(1)     # typical absorption rate
    eta.cl ~ 0.09
    eta.v  ~ 0.04
    add.sd <- 0.2
  })
  model({
    cl <- exp(tcl + eta.cl) * (WT / 70)^0.75 * (1 - 0.005 * (AGE - 50))
    v  <- exp(tv + eta.v) * (WT / 70)
    ka <- exp(tka)
    d/dt(depot) <- -ka * depot
    d/dt(central) <- ka * depot - (cl / v) * central
    cp <- central / v
    cp ~ add(add.sd)
  })
}

fit <- nlmixr2(one_cmt_oral, pk_data, est = "focei")
print(fit)

# Bootstrap confidence intervals via case resampling
library(boot)
boot_fn <- function(data, indices) {
  ids <- unique(data$id)[indices]
  sub <- data[data$id %in% ids, ]
  fit_b <- nlmixr2(one_cmt_oral, sub, est = "focei")
  fixef(fit_b)
}
boot_out <- boot(pk_data, boot_fn, R = 200)
boot.ci(boot_out, type = "perc")
