# mcp_client.py
import asyncio
import json
from typing import List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SimpleMCPClient:
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.connected = False
        self.available_resources = []
        self.available_tools = []
    
    async def connect(self):
        """Establish connection to the MCP server"""
        try:
            # Set up server parameters for stdio connection
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path]
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
            
            # Initialize the connection
            await self.session.initialize()
            self.connected = True
            print(f"✓ Connected to MCP server: {self.server_script_path}")
            
            # Discover capabilities
            await self.discover_capabilities()
            
        except Exception as e:
            print(f"✗ Failed to connect to server: {e}")
            self.connected = False
            raise

    async def discover_capabilities(self):
        """Discover and catalog server resources and tools"""
        print("\n🔍 Discovering Server Capabilities...")
        
        try:
            # Discover available resources
            resources_result = await self.session.list_resources()
            self.available_resources = resources_result.resources
            
            print(f"\n📁 Available Resources ({len(self.available_resources)}):")
            for resource in self.available_resources:
                print(f"  • {resource.name}")
                print(f"    Description: {resource.description}")
                print(f"    URI: {resource.uri}")
                print(f"    Content Type: {resource.mimeType}")
                print()
            
            # Discover available tools
            tools_result = await self.session.list_tools()
            self.available_tools = tools_result.tools
            
            print(f"🛠️  Available Tools ({len(self.available_tools)}):")
            for tool in self.available_tools:
                print(f"  • {tool.name}")
                print(f"    Description: {tool.description}")
                
                # Display input schema in a readable format
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    self._display_tool_schema(tool.inputSchema)
                print()
                
        except Exception as e:
            print(f"✗ Error discovering capabilities: {e}")
            raise

    def _display_tool_schema(self, schema: Dict[str, Any]):
        """Display tool input schema in a user-friendly format"""
        if 'properties' in schema:
            print("    Parameters:")
            for param_name, param_info in schema['properties'].items():
                param_type = param_info.get('type', 'unknown')
                description = param_info.get('description', 'No description')
                default = param_info.get('default', None)
                
                print(f"      - {param_name} ({param_type}): {description}")
                if default is not None:
                    print(f"        Default: {default}")
            
            if 'required' in schema:
                required_fields = schema['required']
                print(f"    Required: {', '.join(required_fields)}")

    def _ensure_connected(self):
        """Verify client is connected to server"""
        if not self.connected:
            raise RuntimeError("❌ Not connected to server. Call connect() first.")

    async def read_users_resource(self) -> List[Dict[str, Any]]:
        """Read and display the users resource"""
        self._ensure_connected()
        
        print("\n📋 Reading Users Resource...")
        try:
            result = await self.session.read_resource("sqlite://users")
            content = result.contents[0].text
            users = json.loads(content)
            
            print(f"📊 Found {len(users)} users in database:")
            print("-" * 50)
            for user in users:
                created_date = user.get('created_at', 'Unknown')
                print(f"  👤 {user['name']}")
                print(f"     Email: {user['email']}")
                print(f"     ID: {user['id']}")
                print(f"     Created: {created_date}")
                print()
                    
            return users
            
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing server response: {e}")
            return []
        except Exception as e:
            print(f"✗ Error reading users resource: {e}")
            return []

    async def query_users(self, filter_term: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Search for users using the query_users tool"""
        self._ensure_connected()
        
        filter_display = f"'{filter_term}'" if filter_term else "none"
        print(f"\n🔍 Querying Users (filter: {filter_display}, limit: {limit})")
        
        try:
            result = await self.session.call_tool(
                "query_users",
                {
                    "filter": filter_term,
                    "limit": limit
                }
            )
            
            # Process and display results
            if result.content and len(result.content) > 0:
                users_data = json.loads(result.content[0].text)
                
                if users_data:
                    print(f"✓ Query returned {len(users_data)} users:")
                    for user in users_data:
                        print(f"  • {user['name']} ({user['email']})")
                else:
                    print("ℹ️  No users match your search criteria")
                    
                return users_data
            else:
                print("⚠️  No response from server")
                return []
                
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing query results: {e}")
            return []
        except Exception as e:
            print(f"✗ Error executing query: {e}")
            return []

    async def add_user(self, name: str, email: str) -> bool:
        """Add a new user using the add_user tool"""
        self._ensure_connected()
        
        # Basic input validation
        if not name or not email:
            print("✗ Both name and email are required")
            return False
        
        if '@' not in email:
            print("✗ Invalid email format")
            return False
        
        print(f"\n➕ Adding User: {name} ({email})")
        
        try:
            result = await self.session.call_tool(
                "add_user",
                {
                    "name": name.strip(),
                    "email": email.strip().lower()
                }
            )
            
            if result.content and len(result.content) > 0:
                response_text = result.content[0].text
                print(f"✓ {response_text}")
                
                # Check if operation was successful
                return "successfully added" in response_text.lower()
            else:
                print("⚠️  No response from server")
                return False
                
        except Exception as e:
            print(f"✗ Error adding user: {e}")
            return False

    def _display_menu(self):
        """Display the main menu options"""
        print("\n" + "-"*30)
        print("📋 Available Commands:")
        print("  1️⃣  List all users (resource)")
        print("  2️⃣  Search users (tool)")
        print("  3️⃣  Add new user (tool)")
        print("  4️⃣  Show server info")
        print("  5️⃣  Exit")
        print("-"*30)

    async def _handle_user_query(self):
        """Handle user search input with validation"""
        print("\n🔍 User Search")
        filter_term = input("Enter search term (name/email, or press Enter for all): ").strip()
        
        limit_input = input("Enter max results (default 10): ").strip()
        try:
            limit = int(limit_input) if limit_input else 10
            limit = max(1, min(limit, 100))  # Clamp between 1 and 100
        except ValueError:
            print("⚠️  Invalid limit, using default (10)")
            limit = 10
        
        await self.query_users(filter_term, limit)

    async def _handle_add_user(self):
        """Handle new user input with validation"""
        print("\n➕ Add New User")
        name = input("Enter full name: ").strip()
        email = input("Enter email address: ").strip()
        
        if name and email:
            success = await self.add_user(name, email)
            if success:
                print("🎉 User added successfully!")
        else:
            print("❌ Both name and email are required")

    async def _display_server_info(self):
        """Display information about the connected server"""
        print(f"\n🔗 Server Information:")
        print(f"  Script: {self.server_script_path}")
        print(f"  Status: {'Connected ✓' if self.connected else 'Disconnected ✗'}")
        print(f"  Resources: {len(self.available_resources)}")
        print(f"  Tools: {len(self.available_tools)}")

    async def interactive_demo(self):
        """Run an interactive demonstration of MCP client capabilities"""
        print("\n" + "="*50)
        print("🚀 MCP Client Interactive Demo")
        print("="*50)
        
        while True:
            self._display_menu()
            choice = input("\n📝 Enter your choice (1-5): ").strip()
            
            try:
                if choice == "1":
                    await self.read_users_resource()
                    
                elif choice == "2":
                    await self._handle_user_query()
                    
                elif choice == "3":
                    await self._handle_add_user()
                    
                elif choice == "4":
                    await self._display_server_info()
                    
                elif choice == "5":
                    print("\n👋 Goodbye!")
                    break
                    
                else:
                    print("❌ Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\n⏹️  Operation cancelled by user")
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
            
            # Pause before showing menu again
            input("\nPress Enter to continue...")

    async def disconnect(self):
        """Safely disconnect from the MCP server"""
        if self.connected:
            try:
                await self.exit_stack.aclose()
                print("✓ Disconnected from MCP server")
            except Exception as e:
                print(f"⚠️  Error during disconnect: {e}")
            finally:
                self.connected = False
                self.available_resources = []
                self.available_tools = []

# Main application entry point
async def main():
    """Main application entry point with proper error handling"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_server_script>")
        print("Example: python mcp_client.py simple_db_server.py")
        sys.exit(1)
    
    server_script = sys.argv[1]
    client = SimpleMCPClient(server_script)
    
    try:
        print("🚀 Starting MCP Client Demo...")
        
        # Connect to server
        await client.connect()
        
        # Run interactive demo
        await client.interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with unexpected error: {e}")
    finally:
        # Ensure cleanup happens
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())