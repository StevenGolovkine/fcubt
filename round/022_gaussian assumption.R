# Test of the Gaussian assumption
# based on T. Gorecki, L. Horvath, and P. Kokoszka. Tests of normality of 
# functional data. International Statistical Review, 88(3):677–697, 2020.

path <- './results/'

# Loop through folders
result <- list()
for(file in list.files(path)){
  print(file)
  if(file == "fcubt_results.pkl") next
  coef <- read.csv(paste0(path, '/', file), header = FALSE)
    
  testMJB <- MVN::mvn(coef, mvnTest = 'mardia')
  testR <- MVN::mvn(coef, mvnTest = 'royston')
  testHZ <- MVN::mvn(coef, mvnTest = 'hz')
  testE <- MVN::mvn(coef, mvnTest = 'energy')
  testDH <- normwhn.test::normality.test1(coef)
  testLV <- normwhn.test::normality.test2(coef)
    
  result[[file]] <- list(
    'MJB' = testMJB$multivariateNormality$`p value`[2],
    'Royston' = testR$multivariateNormality$`p value`,
    'Henze-Zirkler' = testHZ$multivariateNormality$`p value`,
    'Energy' = testE$multivariateNormality$`p value`,
    'Doornik-Hansen' = testDH[1,1],
    'Lobato-Velasco' = testLV[1, 1]
  )
}

saveRDS(results, './results/test_gaussian.rds')
