# ai_assistant.py
import asyncio
import json
import os
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from contextlib import AsyncExitStack

# OpenAI for ChatGPT integration
from openai import OpenAI

# MCP client components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection"""
    name: str
    script_path: str
    description: str
    args: List[str] = None

class MCPConnectionManager:
    """Manages connections to multiple MCP servers"""
    
    def __init__(self):
        self.connections: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.server_tools: Dict[str, List[Dict]] = {}
        
    async def connect_server(self, config: MCPServerConfig) -> bool:
        """Connect to a single MCP server"""
        try:
            logger.info(f"Connecting to MCP server: {config.name}")
            
            # Set up server parameters
            args = [config.script_path]
            if config.args:
                args.extend(config.args)
                
            server_params = StdioServerParameters(
                command="python",
                args=args
            )
            
            # Connect to server
            transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = transport
            
            # Create session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize connection
            await session.initialize()
            self.connections[config.name] = session
            
            # Discover capabilities
            await self._discover_server_capabilities(config.name, session)
            
            logger.info(f"✅ Connected to {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {config.name}: {e}")
            return False
    
    async def _discover_server_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools from a connected server"""
        try:
            tools_result = await session.list_tools()
            self.server_tools[server_name] = []
            
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "server": server_name,
                    "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                }
                self.server_tools[server_name].append(tool_info)
                
        except Exception as e:
            logger.error(f"Error discovering capabilities for {server_name}: {e}")

class MCPFunctionManager:
    """Manages function calling integration between ChatGPT and MCP servers"""
    
    def __init__(self, connection_manager: MCPConnectionManager):
        self.connection_manager = connection_manager
        
    def get_available_functions(self) -> List[Dict]:
        """Generate OpenAI function definitions from MCP tools"""
        functions = []
        
        for server_name, tools in self.connection_manager.server_tools.items():
            for tool in tools:
                # Convert MCP tool schema to OpenAI function format
                function_def = {
                    "name": f"{server_name}_{tool['name']}",
                    "description": f"[{server_name}] {tool['description']}",
                }
                
                # Add parameters if schema exists
                if tool.get('schema'):
                    function_def["parameters"] = tool['schema']
                else:
                    function_def["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                
                functions.append(function_def)
        
        return functions
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool based on ChatGPT function call"""
        try:
            # Parse server and tool name
            if '_' not in function_name:
                return f"Invalid function name format: {function_name}"
            
            # Find the server name by checking against known servers
            server_name = None
            tool_name = None
            
            # Check each known server to see if the function name starts with it
            for known_server in self.connection_manager.connections.keys():
                if function_name.startswith(f"{known_server}_"):
                    server_name = known_server
                    tool_name = function_name[len(known_server) + 1:]  # +1 for the underscore
                    break
            
            if server_name is None:
                return f"Could not determine server for function: {function_name}"
            
            if server_name not in self.connection_manager.connections:
                logger.error(f"Server {server_name} not connected. Available: {list(self.connection_manager.connections.keys())}")
                return f"Server not connected: {server_name}"
            
            session = self.connection_manager.connections[server_name]
            
            # Execute the tool
            logger.info(f"Executing {server_name}.{tool_name} with args: {arguments}")
            result = await session.call_tool(tool_name, arguments)
            
            if result.content and len(result.content) > 0:
                response = result.content[0].text
                #logger.info(f"Tool response: {response[:200]}...")  # Log first 200 chars
                return response
            else:
                logger.error("No response content from server")
                return "No response from server"
                
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error executing function: {str(e)}"

class AIAssistant:
    """Main AI Assistant class that integrates ChatGPT with MCP servers"""
    
    def __init__(self):
        self.connection_manager = MCPConnectionManager()
        self.function_manager = MCPFunctionManager(self.connection_manager)
        self.conversation_history: List[Dict] = []
        self.connected_servers: List[str] = []
        
    async def initialize(self, server_configs: List[MCPServerConfig]):
        """Initialize the assistant by connecting to MCP servers"""
        logger.info("🚀 Initializing AI Assistant...")
        
        for config in server_configs:
            success = await self.connection_manager.connect_server(config)
            if success:
                self.connected_servers.append(config.name)
        
        if not self.connected_servers:
            raise Exception("Failed to connect to any MCP servers")
        
        logger.info(f"✅ Connected to {len(self.connected_servers)} servers: {', '.join(self.connected_servers)}")
        
        # Display available capabilities
        self._display_capabilities()
    
    def _display_capabilities(self):
        """Display available tools"""
        logger.info("\n📊 Available Capabilities:")
        
        for server_name in self.connected_servers:
            tools = self.connection_manager.server_tools.get(server_name, [])
            
            logger.info(f"\n🔧 {server_name} Server:")
            logger.info(f"   Tools: {len(tools)}")
            for tool in tools:
                logger.info(f"   • {tool['name']}: {tool['description']}")
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return AI response"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Get available functions for this conversation
            available_functions = self.function_manager.get_available_functions()
            
            # Create the initial ChatGPT request
            messages = self._build_messages_for_openai()
            
            # Call ChatGPT with function calling enabled
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=available_functions if available_functions else None,
                function_call="auto" if available_functions else None,
                temperature=0.7,
                max_tokens=1500
            )
            
            message = response.choices[0].message
            
            # Check if ChatGPT wants to call a function
            if message.function_call:
                return await self._handle_function_call(message)
            else:
                # Direct response without function calls
                assistant_response = message.content
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                return assistant_response
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    async def _handle_function_call(self, message) -> str:
        """Handle ChatGPT function calls by executing MCP tools"""
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments)
        
        logger.info(f"🔧 ChatGPT calling function: {function_name}")
        logger.info(f"📝 Arguments: {function_args}")
        
        # Debug: Check available connections
        # logger.info(f"🔍 Available connections: {list(self.connection_manager.connections.keys())}")
        
        # Execute the MCP tool
        function_result = await self.function_manager.execute_function(
            function_name, 
            function_args
        )
        
        # Debug: Log the actual result
        # logger.info(f"🔧 Function result: {function_result[:200]}...")
        
        # Add function call and result to conversation
        self.conversation_history.extend([
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": json.dumps(function_args)
                }
            },
            {
                "role": "function",
                "name": function_name,
                "content": function_result
            }
        ])
        
        # Get ChatGPT's response based on the function result
        updated_messages = self._build_messages_for_openai()
        
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=updated_messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        assistant_response = final_response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        return assistant_response
    
    def _build_messages_for_openai(self) -> List[Dict]:
        """Build message history for OpenAI API"""
        messages = [
            {
                "role": "system",
                "content": """You are an intelligent AI assistant with access to real-time data and document processing capabilities. You can:

- Get current weather, news, and stock information
- Search through documents and provide summaries
- Access real-time data to answer questions

When users ask questions that require current information or document access, use the appropriate functions to gather the data before responding. Always provide helpful, accurate, and contextual responses based on the real data you retrieve.

Be conversational and helpful, explaining what you're doing when you call functions to get information."""
            }
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        return messages

class InteractiveAssistant:
    """Interactive interface for the AI Assistant"""
    
    def __init__(self):
        self.assistant = AIAssistant()
    
    async def start(self):
        """Start the interactive assistant"""
        print("🤖 AI Assistant with MCP Integration")
        print("=" * 50)
        
        # Configure MCP servers
        server_configs = [
            MCPServerConfig(
                name="api_gateway",
                script_path="api_gateway_server.py",
                description="Weather, news, and stock data"
            ),
            MCPServerConfig(
                name="documents",
                script_path="document_processing_server.py", 
                description="Document search and analysis",
                args=["./documents"]
            )
        ]
        
        try:
            # Initialize assistant
            await self.assistant.initialize(server_configs)
            
            print("\n✅ AI Assistant ready! Type 'quit' to exit, 'help' for examples.")
            print("💡 Try asking about weather, news, stocks, or searching documents!")
            
            # Start conversation loop
            await self._conversation_loop()
            
        except Exception as e:
            print(f"❌ Failed to initialize assistant: {e}")
    
    async def _conversation_loop(self):
        """Main conversation loop"""
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_examples()
                    continue
                elif user_input.lower() == 'clear':
                    self.assistant.conversation_history.clear()
                    print("🗑️ Conversation history cleared.")
                    continue
                
                # Process message with AI
                print("\n🤖 Assistant: ", end="", flush=True)
                response = await self.assistant.process_message(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_examples(self):
        """Show example queries"""
        print("\n💡 Example queries you can try:")
        print("─" * 40)
        print("🌤️  'What's the weather in Tokyo?'")
        print("📰 'Show me recent news about artificial intelligence'")
        print("📈 'Get the current stock price for Apple'")
        print("📊 'Create a dashboard for New York with tech stocks'")
        print("📄 'Search documents for information about machine learning'")
        print("📋 'Summarize the quarterly report document'")
        print("🔍 'What files do we have about climate change?'")
        print("\n🔧 Commands:")
        print("   'clear' - Clear conversation history")
        print("   'help'  - Show this help message")
        print("   'quit'  - Exit the assistant")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.assistant.connection_manager.exit_stack.aclose()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def main():
    """Main entry point for the AI Assistant"""
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        print("Please set up your .env file with the required API keys.")
        return
    
    interactive_assistant = InteractiveAssistant()
    
    try:
        await interactive_assistant.start()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        await interactive_assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main())