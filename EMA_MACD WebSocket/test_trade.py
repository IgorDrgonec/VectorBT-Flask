import requests

API_URL = "http://127.0.0.1:8080/trade"  # Use the correct address

# Send a test request
#response = requests.post(API_URL, json={"action": "open", "side": "long", "quantity": 0.001})
response = requests.post(API_URL, json={
	"exchange": "BINANCE",
	"ticker": "BTCUSDT.P",
	"stop_loss": 87000,
	"take_profit": 88000,
	"risk_percent": 0.01,
	"strategy": {
		"position_size": "DaviddTech",
		"order_contracts": "DaviddTech",
		"order_price": 87535,
		"market_position": "long"
	}
})

# Print raw response
print("🔹 Status Code:", response.status_code)
print("🔹 Response Text:", response.text)

# Check if it's valid JSON
try:
    json_data = response.json()
    print("✅ JSON Response:", json_data)
except requests.exceptions.JSONDecodeError:
    print("⚠️ Error: Received non-JSON response:", response.text)