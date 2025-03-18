import os
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
import streamlit as st
from langchain.tools import Tool
from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from huggingface_hub import hf_hub_download

# Download the Mistral model from Hugging Face if not already present
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_LOCAL_PATH = os.path.join("models", MODEL_FILENAME)

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_LOCAL_PATH):
    print("Downloading Mistral model from Hugging Face...")
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir="models")
    print("Download completed.")

# Load Local Mistral Model
llm = LlamaCpp(
    model_path=MODEL_LOCAL_PATH,
    n_ctx=2048,
    temperature=0.7
)

# Function to Fetch USD to INR Exchange Rate
def get_usd_to_inr():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data["rates"].get("INR", 0)
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return None

# Function to Fetch Stock Prices and Generate a Graph
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="7d")
    
    if history.empty:
        return f"No data found for {ticker}. Please check the ticker symbol."
    
    latest_price_usd = history["Close"].iloc[-1]
    exchange_rate = get_usd_to_inr()
    latest_price_inr = latest_price_usd * exchange_rate if exchange_rate else "Unavailable"
    
    # Plot the stock price
    plt.figure(figsize=(8, 4))
    plt.plot(history.index, history['Close'], marker='o', linestyle='-')
    plt.title(f"Stock Prices for {ticker}")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.ylabel("Closing Price (USD)")
    plt.grid(True)
    plt.savefig("stock_price.png")
    plt.close()
    
    return {
        "message": f"Stock price for {ticker}: ${latest_price_usd:.2f} (~₹{latest_price_inr:.2f}).",
        "image": "stock_price.png"
    }

# Flask API Setup
app = Flask(__name__)

@app.route('/get_stock_price', methods=['GET'])
def fetch_stock():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "Please provide a stock ticker symbol."})
    
    data = get_stock_price(ticker)
    return jsonify(data)

# Define a LangChain Tool
stock_tool = Tool(
    name="Stock Price Fetcher",
    func=get_stock_price,
    description="Fetches and visualizes stock prices for a given ticker symbol."
)

# Define Memory for the Agent
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the LangChain Agent
agent = initialize_agent(
    tools=[stock_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Streamlit UI
st.title("📈 Stock Price Analyzer")
st.write("Enter a stock symbol to fetch its latest price in USD & INR, along with a trend graph.")

ticker_input = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")
if st.button("Get Stock Price"):
    with st.spinner("Fetching data..."):
        result = get_stock_price(ticker_input.upper())
        st.success(result["message"])
        st.image(result["image"], caption=f"Price Trend of {ticker_input}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
