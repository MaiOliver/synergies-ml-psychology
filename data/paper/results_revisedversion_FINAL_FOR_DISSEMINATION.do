******************************************************************************************
* Neugart M., Richiardi M. (2013). 
* Sequential Teamwork in Competitive Environments: Theory and Evidence from Swimming Data.  
* European Economic Review, vol. 63, pp. 186-205
* Results as in paper
* THE DATA ARE A PROPERTY OF GEOLOGIX AG. AND CAN BE USED ONLY FOR REPLICATION PURPOSES
******************************************************************************************

capture log close
log using "YOUR_LOG_FILE", replace


use "YOUR_FILE_DIRECTORY", clear
use datafinal_dissemination, clear

// choose definition
global major major750

***
* Table 1. Descriptive statistics: distribution of the sample
***

preserve
collapse storder, by(meetid)
sum meetid
restore

count
total _olympic _world _european  _universiade  _panpac  _commonwealth _natchamps
tab _natchamps major1
 
* tab nation  // VARIABLE DROPPED FROM DISSEMINATION FILE
tab gender 
sum age, detail
sum points, detail
tab style
tab schedule
tab round_i 
tab round_r 
tab storder 

tab major1 ${major}, row
tab major2 ${major}, row

***
* Table 2. Average FINA points by starting order
***
// Check whether starting order depends on ability
table storder if athletes_team==4, c(mean points sd points)

char agegroup [omit] "15-30"
char schedule [omit] -1
char style [omit] "100m Freestyle"

xi: reg points i.agegroup major1 major750 major2 female i.schedule i.style storder1 storder2 storder3 storder4 if athletes_team==4, noc vce(cluster athleteid)
test storder1==storder2
test storder2==storder3
test storder3==storder4

xi: reg points i.agegroup major1 major750 major2 female i.schedule i.style storder1 storder2 storder3 storder4, noc vce(cluster athleteid)
test storder1==storder2
test storder2==storder3
test storder3==storder4


***
* Table 4. Descriptive statistics: individual and relay swimming time.
***
sum individual
tab storder,sum(deltatimepct)
ttest deltatimepct==0 if storder==1

bysort gender: sum individual
bysort gender storder: sum deltatimepct 
bysort gender : ttest deltatimepct==0 if storder==1

bysort agegroup : sum individual
bysort agegroup storder: sum deltatimepct 
bysort agegroup : ttest deltatimepct==0 if storder==1

bysort quality : sum individual
bysort quality storder: sum deltatimepct
bysort quality : ttest deltatimepct==0 if storder==1

bysort style : sum individual
bysort style storder: sum deltatimepct
bysort style : ttest deltatimepct==0 if storder==1

bysort major1 : sum individual
bysort major1 storder: sum deltatimepct
bysort major1 : ttest deltatimepct==0 if storder==1

bysort major2 : sum individual
bysort major2 storder: sum deltatimepct
bysort major2 : ttest deltatimepct==0 if storder==1

bysort ${major} : sum individual
bysort ${major} storder: sum deltatimepct
bysort ${major} : ttest deltatimepct==0 if storder==1

bysort schedule : sum individual
bysort schedule storder: sum deltatimepct
bysort schedule : ttest deltatimepct==0 if storder==1

***
* REGRESSIONS
* std.errors are computed with the vce(cluster clustvar) option. This yields the same results as the vce(robust) option 
* which implements the Huber/White/sandwich estimator, since the stata implementation of the 'robust' option with a fixed effects estimator 
* automatically uses -cluster(id)- since some update in version 10. The reason is that the "normal" robust SEs are inconsistent 
* with a FE-estimator (see James H. Stock and Mark Watson, 2008: “Heteroskedasticity-Robust Standard Errors for Fixed Effects Panel Data Regression”, Econometrica 76(1): 155-174).
* Two routines are used for FE estimation: 
* AREG: areg y X, absorb(athleteid) vce(cluster athleteid)
* XTREG: xtset athleteid; xtreg y X, fe vce(cluster athleteid)
* While these various methods yield identical coefficients, the standard errors may differ when Stata’s cluster option is used. 
* When clustering, AREG reports cluster-robust standard errors that reduce the degrees of freedom by the number of fixed effects swept away in the within-group transformation; 
* XTREG reports smaller cluster-robust standard errors because it does not make such an adjustment. 
* XTREG’s approach of not adjusting the degrees of freedom is appropriate when the fixed effects swept away by the within-group transformation 
* are nested within clusters (meaning all the observations for any given group are in the same cluster), as is commonly the case (e.g., firm fixed effects are nested within firm, industry, or state clusters). See Wooldridge (2010, Chapter 20).
* Gormley, Todd A. and Matsa, David A., Common Errors: How to (and Not to) Control for Unobserved Heterogeneity (October 19, 2012). AFA 2013 San Diego Meetings Paper. Available at SSRN: http://ssrn.com/abstract=2023868 or http://dx.doi.org/10.2139/ssrn.2023868
* Moreover, R2 is also not the same: 
* In the AREG procedure, we are estimating coefficients for each of the covariates plus each dummy variable for your groups. 
* In the output of AREG, the overall F test is for the model including the dummy variables (even though we absorbed them and do not see the estimated coefficients). 
* In the XTREG procedure the R2 reported is obtained by only fitting a mean deviated model where the effects of the groups 
* (all of the dummy variables) are assumed to be fixed quantities. So, all of the effects for the groups are simply subtracted 
* out of the model, and no attempt is made to quantify their overall effect on the fit of the model. 
* In the fixed-effects model of XTREG, the overall F test refers only to those terms in which we are interested (the ones that are listed in the output) 
* and does not include the absorbed terms.
* Here we interpret the (average) random intercept, so we want the R2 to consider these terms: this suggests the use of AREG. 
* However, since the fixed effects are nested within clusters we also don't want to adjust the degree of feedom by the number of fixed effects:
* this suggests the use of XTREG
* So, XTREG is used to get the std.errors and the F tests, and AREG is used to get the R2 (here, only XTREG commands are reported)
***

xtset athleteid

***
*regressions on swimmers 1
***
// RMK: the effect of the relay is catched by the constant
* filter: 15-30 age range, freestyle finals only
gen filter = (agegroup=="15-30" & (style=="100m Freestyle" | style=="200m Freestyle" | style=="50m Freestyle") & round_i=="FIN" & round_r=="FIN")
gen filter2 = (agegroup=="15-30" & (style=="100m Freestyle" | style=="200m Freestyle" | style=="50m Freestyle") & (round_i=="FIN"|round_i=="TIM") & (round_r=="FIN"|round_r=="TIM") )

* full sample
xi: reg deltatimepct i.agegroup major1 major750 major2 female i.schedule i.style if storder==1, vce(cluster athleteid)
estimates store ols1, title(ols1)
xi: xtreg deltatimepct major1 major750 major2 female i.schedule i.style if storder==1, fe vce(cluster athleteid)
estimates store fe1, title(fe1)
* + filter2
xi: reg deltatimepct major1 major750 major2 female i.schedule i.style if storder==1 & filter2, vce(cluster athleteid)
estimates store ols2, title(ols2)
xi: xtreg deltatimepct major1 major750 major2 female i.schedule i.style if storder==1 & filter2, fe vce(cluster athleteid)
estimates store fe2, title(fe2)
* + filter2 + major2 (championships finals) PREFERRED SPECIFICATION
xi: reg deltatimepct female i.schedule i.style if storder==1 & filter2 &  major2==1, vce(cluster athleteid) 
estimates store ols3, title(ols3)
xi: xtreg deltatimepct i.schedule i.style if storder==1 & filter2 &  major2==1, fe vce(cluster athleteid)
estimates store fe3, title(fe3)
* + filter2 + ${major}
xi: reg deltatimepct female i.schedule i.style if storder==1 & filter2 &  ${major}==1, vce(cluster athleteid) 
estimates store ols4, title(ols4)
xi: xtreg deltatimepct i.schedule i.style if storder==1 & filter2 &  ${major}==1, fe vce(cluster athleteid)
estimates store fe4, title(fe4)
* + filter2 + major1 (olympics,worlds,europeans,universiades,panpacs,commonwealth)
xi: reg deltatimepct female i.schedule i.style if storder==1 & filter2 & major1==1, vce(cluster athleteid)
estimates store ols5, title(ols5)
xi: xtreg deltatimepct i.schedule i.style if storder==1 & filter2 & major1==1, fe vce(cluster athleteid)
estimates store fe5, title(fe5)

* further results
* + filter2 + major2 (championships finals), first 4 positions
xi: reg deltatimepct female i.schedule i.style if storder==1 & filter2 &  major2==1 & place_i<=4 & place_r<=4, vce(cluster athleteid)
estimates store ols6, title(ols6)
xi: xtreg deltatimepct i.schedule i.style if storder==1 & filter2 &  major2==1 & place_i<=4 & place_r<=4, fe vce(cluster athleteid)
estimates store fe6, title(fe6)
* 100 m Freestyle finals at Olympic events
xi: reg deltatimepct female i.schedule if storder==1 & round_i=="FIN" & round_r=="FIN" & _olympic==1 & style=="100m Freestyle", vce(cluster athleteid)
xi: xtreg deltatimepct i.schedule if storder==1 & round_i=="FIN" & round_r=="FIN" & _olympic==1 & style=="100m Freestyle", fe vce(cluster athleteid)
* separate analysis for daybefore, sameday and dayafter (confidence intervals overlap)
xi: reg deltatimepct female i.style if storder==1 & filter2 &  major2==1 & schedule==-1, vce(cluster athleteid)
estimates store ols7, title(ols7)
xi: xtreg deltatimepct i.style if storder==1 & filter2 &  major2==1 & schedule==-1, fe vce(cluster athleteid)
estimates store fe7, title(fe7)
xi: reg deltatimepct female i.style if storder==1 & filter2 &  major2==1 & schedule==0, vce(cluster athleteid)
estimates store ols8, title(ols8)
xi: xtreg deltatimepct i.style if storder==1 & filter2 &  major2==1 & schedule==0, fe vce(cluster athleteid)
estimates store fe8, title(fe8)
xi: reg deltatimepct female i.style if storder==1 & filter2 &  major2==1 & schedule==+1, vce(cluster athleteid)
estimates store ols9, title(ols9)
xi: xtreg deltatimepct i.style if storder==1 & filter2 &  major2==1 & schedule==+1, fe vce(cluster athleteid)
estimates store fe9, title(fe9)

estout ols1 fe1, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)
estout ols2 fe2 ols3 fe3 ols4 fe4 ols5 fe5, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)
estout ols6 fe6 ols7 fe7 ols8 fe8 ols9 fe9, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)


***
*regressions on swimmers 2-4
***
// RMK: to recover order coefficients add constant to reported coeffs.
* full sample
xi: reg deltatimepct i.agegroup major1 major750 major2 female i.schedule i.style i.storder if storder!=1,noc vce(cluster athleteid)
estimates store ols1_2, title(ols1_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct major1 major750 major2 female i.schedule i.style i.storder if storder!=1, fe vce(cluster athleteid)
estimates store fe1_2, title(fe1_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* + filter2
xi: reg deltatimepct major1 major750 major2 female i.schedule i.style i.storder if storder!=1 & filter2,noc vce(cluster athleteid)
estimates store ols2_2, title(ols2_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct major1 major750 major2 female i.schedule i.style i.storder if storder!=1 & filter2, fe vce(cluster athleteid)
estimates store fe2_2, title(fe2_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* + filter2 + major2 (championships finals) PREFERRED SPECIFICATION
xi: reg deltatimepct female i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1,noc vce(cluster athleteid)
estimates store ols3_2, title(ols3_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1, fe vce(cluster athleteid)
estimates store fe3_2, title(fe3_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* +filter2 + $major (cutoff on points)
xi: reg deltatimepct female i.schedule i.style i.storder if storder!=1 & filter2 &  ${major}==1,noc vce(cluster athleteid)
estimates store ols4_2, title(ols4_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  ${major}==1, fe vce(cluster athleteid)
estimates store fe4_2, title(fe4_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* + filter2 + major1 (olympics,worlds,europeans,universiades,panpacs,commonwealth)
xi: reg deltatimepct female i.schedule i.style i.storder if storder!=1 & filter2 & major1==1,noc vce(cluster athleteid)
estimates store ols5_2, title(ols5_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 & major1==1, fe vce(cluster athleteid)
estimates store fe5_2, title(fe5_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4

* further results
* + filter2 + major2 (championships finals), first 4 positions
xi: reg deltatimepct female i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1 & place_i<=4 & place_r<=4,noc vce(cluster athleteid)
estimates store ols6_2, title(ols6_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1 & place_i<=4 & place_r<=4, fe vce(cluster athleteid)
estimates store fe6_2, title(fe6_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* 100 m Freestyle finals at Olympic events
xi: reg deltatimepct female i.schedule i.storder if storder!=1 & round_i=="FIN" & round_r=="FIN" & _olympic==1 & style=="100m Freestyle",noc
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.storder if storder!=1 & round_i=="FIN" & round_r=="FIN" & _olympic==1 & style=="100m Freestyle", fe
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
* separate analysis for daybefore, sameday and dayafter (confidence intervals overlap)
xi: reg deltatimepct female i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==-1,noc vce(cluster athleteid)
estimates store ols7_2, title(ols7_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==-1, fe vce(cluster athleteid)
estimates store fe7_2, title(fe7_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: reg deltatimepct female i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==0,noc vce(cluster athleteid)
estimates store ols8_2, title(ols8_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==0, fe vce(cluster athleteid)
estimates store fe8_2, title(fe8_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: reg deltatimepct female i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==+1,noc vce(cluster athleteid)
estimates store ols9_2, title(ols9_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if storder!=1 & filter2 &  major2==1 & schedule==+1, fe vce(cluster athleteid)
estimates store fe9_2, title(fe9_2)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4

estout ols1_2 fe1_2, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)
estout ols2_2 fe2_2 ols3_2 fe3_2 ols4_2 fe4_2 ols5_2 fe5_2, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)
estout ols6_2 fe6_2 ols7_2 fe7_2 ols8_2 fe8_2 ols9_2 fe9_2, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)

***
*regressions on swimmers 4 (to compute reaction time advantage)
***
* absolute diff preferred specification, different lengths 
xi: reg deltatime female i.schedule if storder==4 & filter2 &  major2==1 & style=="50m Freestyle", vce(cluster athleteid)
xi: xtreg deltatime i.schedule if storder==4 & filter2 &  major2==1 & style=="50m Freestyle", fe vce(cluster athleteid)
xi: reg deltatime female i.schedule if storder==4 & filter2 &  major2==1 & style=="100m Freestyle", vce(cluster athleteid)
xi: xtreg deltatime i.schedule if storder==4 & filter2 &  major2==1 & style=="100m Freestyle", fe vce(cluster athleteid)
xi: reg deltatime female i.schedule if storder==4 & filter2 &  major2==1 & style=="200m Freestyle", vce(cluster athleteid)
xi: xtreg deltatime i.schedule if storder==4 & filter2 &  major2==1 & style=="200m Freestyle", fe vce(cluster athleteid)

***
*regression on swimmers 1-4
***
* + filter2 + major2 (championships finals) PREFERRED SPECIFICATION
xi: reg deltatimepct female i.schedule i.style i.storder if filter2 &  major2==1, vce(cluster athleteid)
estimates store ols3_3, title(ols3_3)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4
xi: xtreg deltatimepct i.schedule i.style i.storder if filter2 &  major2==1, fe vce(cluster athleteid)
estimates store fe3_3, title(fe3_3)
test _Istorder_2 = _Istorder_3
test _Istorder_3 = _Istorder_4

estout ols3_2 fe3_2, cells(b(star fmt(2))) starlevels(* 0.10 ** 0.05 *** 0.01) stats(r2 N)

***
* quantification of reaction time
***
/*
sum deltatime if storder==4
table major2 if storder==4, c(mean deltatime sd deltatime)
table major750 if storder==4, c(mean deltatime sd deltatime)
table major1 if storder==4, c(mean deltatime sd deltatime)
*/
xi: reg deltatime i.agegroup female i.style if storder==4, vce(cluster athleteid)
*xi: xtreg deltatimepct female i.style if storder==1, fe vce(cluster athleteid)


***
*difference in swimming time by final placements
***
// Copy and paste all the following collapsed datasets in the file 'Impact of effect - feb2013.xls' (sheet 'data').
// Then, update the pivot (sheet 'pivot') and select the appropriate filters

preserve
keep if place_r <= 20
collapse relay, by(gender style place_r filter2 major2)
restore
 
preserve
keep if place_r <= 20
collapse relay, by(gender style place_r filter2)
restore
 
preserve
keep if place_r <= 20
collapse relay, by(gender style place_r)
restore

log close




