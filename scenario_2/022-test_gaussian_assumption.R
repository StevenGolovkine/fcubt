# Test of the Gaussian assumption
# based on T. Gorecki, L. Horvath, and P. Kokoszka. Tests of normality of 
# functional data. International Statistical Review, 88(3):677–697, 2020.

coef <- read.csv('./scenario_2/results/coef.csv', header = FALSE)

test <- MVN::mvn(coef, mvnTest = 'mardia')
test <- MVN::mvn(coef, mvnTest = 'royston')
test <- MVN::mvn(coef, mvnTest = 'hz')
test <- MVN::mvn(coef, mvnTest = 'energy')
test <- normwhn.test::normality.test1(coef)
test <- normwhn.test::normality.test2(coef)
