# api_gateway_server.py
import aiohttp
import asyncio
import json
import os
import concurrent.futures
from typing import Dict, List, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import MCP components using the correct structure
from mcp.server.fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("api-gateway-server")

class APIGatewayHandler:
    def __init__(self):
        self.api_configs = {
            "weather": {
                "base_url": "https://api.openweathermap.org/data/2.5",
                "auth_type": "api_key",
                "auth_key": os.getenv("WEATHER_API_KEY")
            },
            "news": {
                "base_url": "https://newsapi.org/v2",
                "auth_type": "api_key",
                "auth_key": os.getenv("NEWS_API_KEY")
            },
            "stocks": {
                "base_url": "https://api.finnhub.io/api/v1",
                "auth_type": "token",
                "auth_key": os.getenv("FINNHUB_TOKEN")
            }
        }
        
        # Validate API keys
        missing_keys = []
        for api_name, config in self.api_configs.items():
            if not config["auth_key"]:
                missing_keys.append(api_name)
        
        if missing_keys:
            print(f"Warning: Missing API keys for: {', '.join(missing_keys)}")
            print("Some features may not work. Please check your .env file.")
    
    async def _make_api_request(self, api_name: str, endpoint: str, params: Dict = None):
        """Generic method to make API requests with proper authentication"""
        config = self.api_configs[api_name]
        url = f"{config['base_url']}/{endpoint}"
        
        headers = {}
        if config["auth_type"] == "api_key":
            params = params or {}
            # Different APIs use different parameter names for API keys
            if api_name == "weather":
                params["appid"] = config["auth_key"]
            else:
                params["apiKey"] = config["auth_key"]
        elif config["auth_type"] == "token":
            headers["X-Finnhub-Token"] = config["auth_key"]
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")

# Create handler instance
api_handler = APIGatewayHandler()

@mcp.tool()
def get_weather(city: str, units: str = "metric") -> str:
    """Get current weather for a location
    
    Args:
        city: City name
        units: Temperature units (metric or imperial)
    """
    try:
        # Use asyncio.run() which handles the event loop properly
        async def _get_weather():
            try:
                data = await api_handler._make_api_request(
                    "weather",
                    "weather", 
                    {"q": city, "units": units}
                )
                
                weather_info = {
                    "location": data["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "units": units
                }
                
                return json.dumps(weather_info, indent=2)
                
            except Exception as e:
                return f"Error fetching weather data: {str(e)}"
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to use run_in_executor or similar
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_weather())
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(_get_weather())
            
    except Exception as e:
        return f"Error executing weather tool: {str(e)}"

@mcp.tool()
def get_news(query: str = "", language: str = "en", page_size: int = 10) -> str:
    """Get latest news articles
    
    Args:
        query: Search term for news articles
        language: Language code for articles
        page_size: Number of articles to return
    """
    try:
        async def _get_news():
            try:
                params = {
                    "language": language,
                    "pageSize": page_size
                }
                
                # Use different endpoints based on whether we have a query
                if query:
                    params["q"] = query
                    endpoint = "everything"
                else:
                    endpoint = "top-headlines"
                    params["country"] = "us"  # Default to US headlines
                
                data = await api_handler._make_api_request("news", endpoint, params)
                
                articles = []
                for article in data.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "Unknown")
                    })
                
                return json.dumps({
                    "total_results": len(articles),
                    "articles": articles
                }, indent=2)
                
            except Exception as e:
                return f"Error fetching news data: {str(e)}"
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_news())
                return future.result()
        except RuntimeError:
            return asyncio.run(_get_news())
            
    except Exception as e:
        return f"Error executing news tool: {str(e)}"

@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """Get current stock price
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL)
    """
    try:
        async def _get_stock_price():
            try:
                symbol_upper = symbol.upper()
                data = await api_handler._make_api_request(
                    "stocks", 
                    "quote",
                    {"symbol": symbol_upper}
                )
                
                stock_info = {
                    "symbol": symbol_upper,
                    "current_price": data.get("c", 0),
                    "change": data.get("d", 0),
                    "percent_change": data.get("dp", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0),
                    "open": data.get("o", 0),
                    "previous_close": data.get("pc", 0)
                }
                
                return json.dumps(stock_info, indent=2)
                
            except Exception as e:
                return f"Error fetching stock data: {str(e)}"
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_stock_price())
                return future.result()
        except RuntimeError:
            return asyncio.run(_get_stock_price())
            
    except Exception as e:
        return f"Error executing stock tool: {str(e)}"

@mcp.tool()
def aggregate_dashboard(location: str, stock_symbols: List[str] = None, news_topics: List[str] = None) -> str:
    """Get aggregated data for a business dashboard
    
    Args:
        location: City for weather data
        stock_symbols: List of stock symbols to track
        news_topics: List of news topics to search
    """
    if stock_symbols is None:
        stock_symbols = ["AAPL", "GOOGL", "MSFT"]
    if news_topics is None:
        news_topics = ["technology", "business"]
    
    try:
        async def _aggregate_dashboard():
            dashboard_data = {}
            
            # Fetch weather data
            try:
                weather_data = await api_handler._make_api_request(
                    "weather",
                    "weather",
                    {"q": location, "units": "metric"}
                )
                dashboard_data["weather"] = {
                    "location": weather_data["name"],
                    "temperature": weather_data["main"]["temp"],
                    "description": weather_data["weather"][0]["description"],
                    "humidity": weather_data["main"]["humidity"],
                    "wind_speed": weather_data["wind"]["speed"]
                }
            except Exception as e:
                dashboard_data["weather"] = {"error": str(e)}
            
            # Fetch stock data concurrently
            dashboard_data["stocks"] = {}
            stock_tasks = []
            for symbol in stock_symbols:
                task = api_handler._make_api_request("stocks", "quote", {"symbol": symbol.upper()})
                stock_tasks.append(task)
            
            try:
                stock_results = await asyncio.gather(*stock_tasks, return_exceptions=True)
                for symbol, result in zip(stock_symbols, stock_results):
                    if isinstance(result, Exception):
                        dashboard_data["stocks"][symbol] = {"error": str(result)}
                    else:
                        dashboard_data["stocks"][symbol] = {
                            "symbol": symbol.upper(),
                            "current_price": result.get("c", 0),
                            "change": result.get("d", 0),
                            "percent_change": result.get("dp", 0)
                        }
            except Exception as e:
                dashboard_data["stocks"] = {"error": str(e)}
            
            # Fetch news data
            dashboard_data["news"] = {}
            for topic in news_topics:
                try:
                    news_data = await api_handler._make_api_request(
                        "news", 
                        "everything", 
                        {"q": topic, "pageSize": 5, "language": "en"}
                    )
                    articles = []
                    for article in news_data.get("articles", [])[:5]:
                        articles.append({
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "url": article.get("url", ""),
                            "source": article.get("source", {}).get("name", "Unknown")
                        })
                    dashboard_data["news"][topic] = {
                        "total_results": len(articles),
                        "articles": articles
                    }
                except Exception as e:
                    dashboard_data["news"][topic] = {"error": str(e)}
            
            return json.dumps(dashboard_data, indent=2)
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _aggregate_dashboard())
                return future.result()
        except RuntimeError:
            return asyncio.run(_aggregate_dashboard())
            
    except Exception as e:
        return f"Error executing dashboard tool: {str(e)}"

# Server startup
if __name__ == "__main__":
    mcp.run()