from reflection_agent import ReflectionAgent

model = "gemma4:latest"

generation_system_prompt = "You are a financial advisor. You analyze the stock market and provide suggestions to investors"
"in which stock is ready to buy."
"You have access to the stock market data and you can analyze it to provide insights to investors."
"Keep in mind the provided information must be true, accurate and up to date. Don't make up any information. If you don't know the answer, say you don't know."
"Don't use emojis in your report. Keep report under 200 character and to the point."

stock_details = {
  "stock_name": "Ather Energy Ltd",
  "ticker": "NSE: ATHERENERG",
  "last_updated": "2026-04-22",
  "current_price": 896.00,
  "market_cap_inr_cr": 34651,
  "52_week_high": 948.45,
  "52_week_low": 287.30,
  "six_month_performance": {
    "trend": "Upward",
    "notes": "Strong recovery from IPO lows, driven by high demand for Rizta scooters and 50% YoY volume growth as of Dec 2025"
  },
  "financial_highlights_q3_fy26": {
    "revenue_growth": "50% YoY",
    "market_share": "18.8%",
    "ebitda_margin": "-3% (Improving)",
    "propack_attachment_rate": "91%"
  },
  "operational_updates": {
    "retail_centres": "351 (Mar 2026), Target 700 by end of FY26",
    "main_models": ["450X", "Rizta"]
  },
  "technical_indicators": {
    "sma_50": 749.8,
    "sma_200": 610.7,
    "outlook": "Bullish, High Volume"
  }
}

user_msg = f"Let me know when what will be right price to buy Ather stocks. Additional stock details in JSON format: \n {stock_details}"

reflection_system_prompt =  "you are SEBI-registered financial advisor. "
"You tasked with generating critique and recommendations "
"for the user's analysis.",

def main():
  ReflectionAgent(model=model).run(
    user_msg,
    generation_system_prompt,
    reflection_system_prompt,
    verbose=1,
    n_steps=3
  )
  
if __name__ == "__main__":
  main()
