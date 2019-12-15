import calculate_metrics
pred2 = ["great product", "excellent", "great dog food", "very tasty", "not bad", "a little beverage", "great tea", "great product", "great product"]

pred = ["good", "great product", "my dog loves this food", "good", "a good tea", "not bad but not too sweet", "not as good as the", "not bad"]

ref = ["delicious addictive", "great product", "design for teeth and health", "best milk chocolate not creamy great shape", "very weak flavor great", "tasty juice is not fresh", "really enjoyed healthy for you", "good quality spicy", "great flavor branding snack sized bag addicting delicious tea"]

lda_predict = ["disappointed flavor chocolate notes especially weak milk thickens flavor still disappoints worth try never buy use left gone time thanks small cans"]
lda_gold = ["best milk chocolate not creamy great shape"]
calculate_metrics.compute(lda_gold, lda_predict)
#calculate_metrics.compute(ref, pred2)
    
