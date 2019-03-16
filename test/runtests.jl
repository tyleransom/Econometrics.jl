using Test, Econometrics

# Balanced Panel Data
data = dataset("Ecdat", "Crime")
reg = fit(EconometricModel,
          @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
          data)
xtreg_be = fit(EconometricModel,
               @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + between(County)),
               data)
xtreg_fe = fit(EconometricModel,
               @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
               data)
xtreg_fe_i = fit(EconometricModel,
                 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
                 data)
xtreg = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + PID(County) + TID(Year)),
            data)
ivreg = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
            data)
ivreg_be = fit(EconometricModel,
               @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + between(County)),
               data)
ivreg_fe = fit(EconometricModel,
               @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County)),
               data)
ivreg_fe_i = fit(EconometricModel,
                 @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)),
                 data)
ivreg = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + PID(County) + TID(Year)),
            data)
# Ordinal Logistic Regression
data = dataset("Ecdat", "Mathlevel")