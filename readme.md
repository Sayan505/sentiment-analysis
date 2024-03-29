### Example API Call:  

URI (POST): `<baseurl>/api/predict_sentiment`  

Request JSON:
```json
{
	"query": "This cat looks very cute!"
}
```

Response JSON:
```json
{
	"admiration": 100.0,
	"amusement": 0.0,
	"anger": 0.0,
	"annoyance": 0.0,
	"approval": 0.0,
	"caring": 0.0,
	"confusion": 0.0,
	"curiosity": 0.0,
	"desire": 0.0,
	"disappointment": 0.0,
	"disapproval": 0.0,
	"disgust": 0.0,
	"embarrassment": 0.0,
	"excitement": 0.0,
	"fear": 0.0,
	"gratitude": 0.0,
	"grief": 0.0,
	"joy": 0.0,
	"love": 0.0,
	"nervousness": 0.0,
	"optimism": 0.0,
	"pride": 0.0,
	"realization": 0.0,
	"relief": 0.0,
	"remorse": 0.0,
	"sadness": 0.0,
	"surprise": 0.0,
	"neutral": 0.0
}
```

### Status Page:  

URI (GET): `<baseurl>/api/status`  

Response JSON:
```json
{
	"status": "ok",
	"labels": "../model/emotions.csv",
	"vectorizer": "../model/vectorizer.pickle",
	"model": "../model/DTCModel.pickle"
}
```
