from collections import defaultdict

def ensemble(weighted_classifiers):
	result = defaultdict(float)
	for predictor, weight in weighted_classifiers:	
		for key in predictor:
			result[key] += predictor[key] * weight
	return result