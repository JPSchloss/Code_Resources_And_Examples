# interactive_test_client.py
import asyncio
import json
import subprocess
import sys
import time
import os
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack

class InteractiveAPIGatewayTester:
    def __init__(self):
        self.server_process = None
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.connected = False
        self.available_tools = []
        
    async def start_server(self):
        """Start the API Gateway server as a subprocess"""
        print("🚀 Starting API Gateway Server...")
        try:
            # Start server process
            self.server_process = await asyncio.create_subprocess_exec(
                sys.executable, "api_gateway_server.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Give server time to start
            await asyncio.sleep(2)
            
            # Check if server is still running
            if self.server_process.returncode is not None:
                stderr = await self.server_process.stderr.read()
                raise Exception(f"Server failed to start: {stderr.decode()}")
            
            print("✅ Server started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    async def connect_to_server(self):
        """Connect to the running server using MCP client"""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            from mcp import StdioServerParameters
            
            # Set up connection parameters
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["api_gateway_server.py"]
            )
            
            # Connect to server
            transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = transport
            
            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize connection
            await self.session.initialize()
            self.connected = True
            print("✅ Connected to MCP server!")
            
            # Discover available tools
            await self.discover_tools()
            return True
            
        except ImportError:
            print("❌ MCP client not available. Using direct function calls instead.")
            return await self.setup_direct_mode()
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def setup_direct_mode(self):
        """Set up direct mode when MCP client is not available"""
        try:
            # Import the server module directly
            import api_gateway_server
            self.api_module = api_gateway_server
            self.connected = True
            
            # Manually define available tools
            self.available_tools = [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": ["city", "units (optional)"]
                },
                {
                    "name": "get_news", 
                    "description": "Get latest news articles",
                    "parameters": ["query (optional)", "language (optional)", "page_size (optional)"]
                },
                {
                    "name": "get_stock_price",
                    "description": "Get current stock price", 
                    "parameters": ["symbol"]
                },
                {
                    "name": "aggregate_dashboard",
                    "description": "Get aggregated dashboard data",
                    "parameters": ["location", "stock_symbols (optional)", "news_topics (optional)"]
                }
            ]
            
            print("✅ Direct mode enabled!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set up direct mode: {e}")
            return False
    
    async def discover_tools(self):
        """Discover available tools from the server"""
        try:
            tools_result = await self.session.list_tools()
            self.available_tools = []
            
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                }
                self.available_tools.append(tool_info)
            
            print(f"🔍 Discovered {len(self.available_tools)} tools")
            
        except Exception as e:
            print(f"⚠️ Error discovering tools: {e}")
    
    def display_tools(self):
        """Display available tools"""
        print("\n📊 Available Tools:")
        print("=" * 50)
        
        for i, tool in enumerate(self.available_tools, 1):
            print(f"{i}. {tool['name']}")
            print(f"   📝 {tool['description']}")
            
            if 'schema' in tool and tool['schema']:
                self.display_schema(tool['schema'])
            elif 'parameters' in tool:
                print(f"   📋 Parameters: {', '.join(tool['parameters'])}")
            print()
    
    def display_schema(self, schema):
        """Display tool schema in a readable format"""
        if 'properties' in schema:
            params = []
            required = schema.get('required', [])
            
            for param_name, param_info in schema['properties'].items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', '')
                is_required = param_name in required
                
                param_str = f"{param_name} ({param_type})"
                if not is_required:
                    param_str += " [optional]"
                if param_desc:
                    param_str += f" - {param_desc}"
                
                params.append(param_str)
            
            if params:
                print(f"   📋 Parameters:")
                for param in params:
                    print(f"      • {param}")
    
    async def call_tool_mcp(self, tool_name: str, arguments: Dict[str, Any]):
        """Call tool using MCP protocol"""
        try:
            result = await self.session.call_tool(tool_name, arguments)
            if result.content and len(result.content) > 0:
                return result.content[0].text
            else:
                return "No response from server"
        except Exception as e:
            return f"Error calling tool: {e}"
    
    def call_tool_direct(self, tool_name: str, **kwargs):
        """Call tool directly"""
        try:
            if tool_name == "get_weather":
                return self.api_module.get_weather(
                    kwargs.get('city', ''),
                    kwargs.get('units', 'metric')
                )
            elif tool_name == "get_news":
                return self.api_module.get_news(
                    kwargs.get('query', ''),
                    kwargs.get('language', 'en'),
                    kwargs.get('page_size', 10)
                )
            elif tool_name == "get_stock_price":
                return self.api_module.get_stock_price(kwargs.get('symbol', ''))
            elif tool_name == "aggregate_dashboard":
                return self.api_module.aggregate_dashboard(
                    kwargs.get('location', ''),
                    kwargs.get('stock_symbols'),
                    kwargs.get('news_topics')
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error calling tool: {e}"
    
    async def test_weather_tool(self):
        """Interactive test for weather tool"""
        print("\n🌤️ Weather Tool Test")
        print("-" * 30)
        
        city = input("Enter city name: ").strip()
        if not city:
            print("❌ City name is required")
            return
        
        units = input("Enter units (metric/imperial) [metric]: ").strip() or "metric"
        
        print(f"\n🔄 Getting weather for {city}...")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("get_weather", {
                "city": city,
                "units": units
            })
        else:
            result = self.call_tool_direct("get_weather", city=city, units=units)
        
        print("\n📊 Result:")
        try:
            # Try to format JSON nicely
            data = json.loads(result)
            print(json.dumps(data, indent=2))
        except:
            print(result)
    
    async def test_news_tool(self):
        """Interactive test for news tool"""
        print("\n📰 News Tool Test")
        print("-" * 30)
        
        query = input("Enter search query (optional): ").strip()
        language = input("Enter language [en]: ").strip() or "en"
        page_size_str = input("Enter number of articles [5]: ").strip() or "5"
        
        try:
            page_size = int(page_size_str)
        except ValueError:
            page_size = 5
        
        print(f"\n🔄 Getting news...")
        if query:
            print(f"   Query: {query}")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("get_news", {
                "query": query,
                "language": language,
                "page_size": page_size
            })
        else:
            result = self.call_tool_direct("get_news", 
                                         query=query, 
                                         language=language, 
                                         page_size=page_size)
        
        print("\n📊 Result:")
        try:
            data = json.loads(result)
            if "articles" in data:
                print(f"Found {data.get('total_results', 0)} articles:")
                for i, article in enumerate(data["articles"][:3], 1):
                    print(f"\n{i}. {article.get('title', 'No title')}")
                    print(f"   Source: {article.get('source', 'Unknown')}")
                    if article.get('description'):
                        print(f"   {article['description'][:100]}...")
            else:
                print(json.dumps(data, indent=2))
        except:
            print(result)
    
    async def test_stock_tool(self):
        """Interactive test for stock tool"""
        print("\n📈 Stock Tool Test")
        print("-" * 30)
        
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        if not symbol:
            print("❌ Stock symbol is required")
            return
        
        print(f"\n🔄 Getting stock price for {symbol}...")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("get_stock_price", {"symbol": symbol})
        else:
            result = self.call_tool_direct("get_stock_price", symbol=symbol)
        
        print("\n📊 Result:")
        try:
            data = json.loads(result)
            if "current_price" in data:
                print(f"Symbol: {data.get('symbol', symbol)}")
                print(f"Current Price: ${data.get('current_price', 'N/A')}")
                print(f"Change: {data.get('change', 'N/A')} ({data.get('percent_change', 'N/A')}%)")
                print(f"High: ${data.get('high', 'N/A')}")
                print(f"Low: ${data.get('low', 'N/A')}")
            else:
                print(json.dumps(data, indent=2))
        except:
            print(result)
    
    async def test_dashboard_tool(self):
        """Interactive test for dashboard tool"""
        print("\n📊 Dashboard Tool Test")
        print("-" * 30)
        
        location = input("Enter location for weather: ").strip()
        if not location:
            print("❌ Location is required")
            return
        
        stocks_input = input("Enter stock symbols (comma-separated) [AAPL,GOOGL]: ").strip()
        news_input = input("Enter news topics (comma-separated) [technology,business]: ").strip()
        
        stock_symbols = None
        if stocks_input:
            stock_symbols = [s.strip().upper() for s in stocks_input.split(',')]
        
        news_topics = None
        if news_input:
            news_topics = [t.strip() for t in news_input.split(',')]
        
        print(f"\n🔄 Generating dashboard for {location}...")
        
        if hasattr(self, 'session') and self.session:
            params = {"location": location}
            if stock_symbols:
                params["stock_symbols"] = stock_symbols
            if news_topics:
                params["news_topics"] = news_topics
            result = await self.call_tool_mcp("aggregate_dashboard", params)
        else:
            result = self.call_tool_direct("aggregate_dashboard",
                                         location=location,
                                         stock_symbols=stock_symbols,
                                         news_topics=news_topics)
        
        print("\n📊 Dashboard Result:")
        try:
            data = json.loads(result)
            
            # Display weather
            if "weather" in data:
                weather = data["weather"]
                if "error" not in weather:
                    print(f"\n🌤️ Weather in {weather.get('location', location)}:")
                    print(f"   Temperature: {weather.get('temperature', 'N/A')}°")
                    print(f"   Description: {weather.get('description', 'N/A')}")
                else:
                    print(f"\n⚠️ Weather Error: {weather['error']}")
            
            # Display stocks
            if "stocks" in data:
                print(f"\n📈 Stock Prices:")
                for symbol, stock_data in data["stocks"].items():
                    if "error" not in stock_data:
                        price = stock_data.get('current_price', 'N/A')
                        change = stock_data.get('percent_change', 'N/A')
                        print(f"   {symbol}: ${price} ({change}%)")
                    else:
                        print(f"   {symbol}: Error - {stock_data['error']}")
            
            # Display news summary
            if "news" in data:
                print(f"\n📰 News Summary:")
                for topic, news_data in data["news"].items():
                    if "error" not in news_data:
                        count = news_data.get('total_results', 0)
                        print(f"   {topic}: {count} articles found")
                    else:
                        print(f"   {topic}: Error - {news_data['error']}")
        except:
            print(result)
    
    async def check_api_keys(self):
        """Check API key configuration"""
        print("\n🔑 API Key Status Check")
        print("-" * 30)
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("⚠️ python-dotenv not installed")
        
        keys = {
            "Weather API": os.getenv("WEATHER_API_KEY"),
            "News API": os.getenv("NEWS_API_KEY"),
            "Stock API": os.getenv("FINNHUB_TOKEN")
        }
        
        configured_count = 0
        for name, key in keys.items():
            if key and not key.startswith("your_"):
                print(f"✅ {name}: Configured")
                configured_count += 1
            else:
                print(f"❌ {name}: Not configured")
        
        print(f"\n📊 Status: {configured_count}/3 API keys configured")
        
        if configured_count == 0:
            print("\n⚠️ No API keys found. Tools will return errors.")
            print("\n📋 Get your free API keys from:")
            print("   🌤️ Weather: https://openweathermap.org/api")
            print("   📰 News: https://newsapi.org/")
            print("   📈 Stocks: https://finnhub.io/")
        elif configured_count < 3:
            print(f"\n⚠️ Only {configured_count} API keys configured. Some tools won't work.")
    
    async def run_interactive_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "=" * 60)
            print("🔧 Interactive API Gateway Tester")
            print("=" * 60)
            print("1. 🌤️  Test Weather Tool")
            print("2. 📰 Test News Tool")
            print("3. 📈 Test Stock Tool")
            print("4. 📊 Test Dashboard Tool")
            print("5. 🔑 Check API Key Status")
            print("6. 📋 Show Available Tools")
            print("7. 🚪 Exit")
            
            choice = input("\nSelect an option (1-7): ").strip()
            
            try:
                if choice == "1":
                    await self.test_weather_tool()
                elif choice == "2":
                    await self.test_news_tool()
                elif choice == "3":
                    await self.test_stock_tool()
                elif choice == "4":
                    await self.test_dashboard_tool()
                elif choice == "5":
                    await self.check_api_keys()
                elif choice == "6":
                    self.display_tools()
                elif choice == "7":
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-7.")
            
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation cancelled")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            if choice != "7":
                input("\nPress Enter to continue...")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.connected:
                await self.exit_stack.aclose()
            
            if self.server_process and self.server_process.returncode is None:
                self.server_process.terminate()
                await self.server_process.wait()
                
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def main():
    """Main function"""
    tester = InteractiveAPIGatewayTester()
    
    try:
        print("🚀 Starting Interactive API Gateway Tester")
        print("=" * 50)
        
        # Try to connect (this will use direct mode if MCP client unavailable)
        if await tester.connect_to_server():
            await tester.run_interactive_menu()
        else:
            print("❌ Failed to initialize tester")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Tester interrupted by user")
    except Exception as e:
        print(f"\n💥 Tester failed: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())