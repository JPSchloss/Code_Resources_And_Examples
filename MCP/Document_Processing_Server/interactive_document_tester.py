# interactive_document_tester.py
import asyncio
import json
import subprocess
import sys
import time
import os
import tempfile
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
from pathlib import Path

class InteractiveDocumentTester:
    def __init__(self, test_directory: str = None):
        self.test_directory = test_directory
        self.server_process = None
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.connected = False
        self.available_tools = []
        self.available_resources = []
        
    async def setup_test_documents(self, directory: str) -> bool:
        """Create sample documents for testing"""
        try:
            test_dir = Path(directory)
            test_dir.mkdir(exist_ok=True)
            
            # Create sample text file
            (test_dir / "sample.txt").write_text("""
Machine Learning and Artificial Intelligence
============================================

This document contains information about machine learning and artificial intelligence.
Machine learning is a subset of artificial intelligence that focuses on algorithms
that can learn from data without being explicitly programmed.

Key concepts in machine learning include:
- Supervised learning
- Unsupervised learning  
- Reinforcement learning
- Neural networks
- Deep learning

Applications of AI include natural language processing, computer vision,
robotics, and autonomous systems.
""")
            
            # Create sample markdown file
            (test_dir / "project_notes.md").write_text("""# Project Documentation

## Overview
This project implements a document processing system using Python.

## Features
- Text extraction from multiple file formats
- Document search capabilities
- Summarization algorithms
- File management

## Technologies Used
- Python 3.8+
- FastMCP framework
- Document processing libraries

## Installation
```bash
pip install fastmcp PyPDF2 python-docx
```

## Usage
The system provides tools for processing documents and extracting insights.
""")
            
            # Create sample JSON file
            (test_dir / "config.json").write_text(json.dumps({
                "application": "Document Processor",
                "version": "1.0.0",
                "settings": {
                    "max_file_size": "10MB",
                    "supported_formats": ["txt", "pdf", "docx", "md", "json", "csv"],
                    "search_options": {
                        "case_sensitive": False,
                        "max_results": 50
                    }
                },
                "features": [
                    "text_extraction",
                    "document_search", 
                    "summarization",
                    "file_listing"
                ]
            }, indent=2))
            
            # Create sample CSV file
            (test_dir / "data.csv").write_text("""name,department,role,years_experience
Alice Johnson,Engineering,Senior Developer,5
Bob Smith,Marketing,Marketing Manager,7
Carol Davis,Engineering,Tech Lead,8
David Wilson,Sales,Account Manager,3
Eva Brown,Engineering,Software Engineer,2
Frank Miller,HR,HR Specialist,4
""")
            
            # Create a subdirectory with more files
            sub_dir = test_dir / "reports"
            sub_dir.mkdir(exist_ok=True)
            
            (sub_dir / "quarterly_report.txt").write_text("""
Quarterly Business Report
========================

Executive Summary:
Our Q3 performance shows strong growth in the technology sector.
Revenue increased by 15% compared to the previous quarter.

Key Metrics:
- Total Revenue: $2.5M
- Customer Acquisition: 150 new customers
- Employee Satisfaction: 8.5/10
- Technology Adoption: 85% of processes automated

Challenges:
- Supply chain disruptions
- Increased competition
- Remote work coordination

Opportunities:
- AI integration possibilities
- Market expansion in Asia
- Partnership opportunities
""")
            
            (sub_dir / "meeting_notes.md").write_text("""# Team Meeting Notes

## Date: 2024-01-15
## Attendees: Engineering Team

### Agenda Items
1. Document processing system review
2. Performance optimization discussion  
3. Future feature planning

### Key Decisions
- Implement new search algorithms
- Upgrade document parsing libraries
- Add support for more file formats

### Action Items
- [ ] Research new PDF parsing options
- [ ] Benchmark current performance
- [ ] Design new API endpoints
- [ ] Update documentation

### Next Meeting: 2024-01-22
""")
            
            print(f"✅ Created {len(list(test_dir.rglob('*')))} test files in {directory}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating test documents: {e}")
            return False
    
    async def connect_to_server(self):
        """Connect to the running server using MCP client"""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            from mcp import StdioServerParameters
            
            if not self.test_directory:
                print("❌ No test directory specified")
                return False
            
            # Set up connection parameters
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["document_processing_server.py", self.test_directory]
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
            
            # Discover available tools and resources
            await self.discover_capabilities()
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
            sys.path.insert(0, '.')
            
            # We need to simulate the server setup
            import document_processing_server
            
            # Initialize the global handler
            from document_processing_server import DocumentProcessingHandler
            document_processing_server.doc_handler = DocumentProcessingHandler(self.test_directory)
            
            self.doc_module = document_processing_server
            self.connected = True
            
            # Manually define available tools
            self.available_tools = [
                {
                    "name": "list_files",
                    "description": "List all files of a specific type",
                    "parameters": ["file_type (optional)", "include_subdirs (optional)"]
                },
                {
                    "name": "search_documents", 
                    "description": "Search for text across documents",
                    "parameters": ["query", "file_types (optional)", "case_sensitive (optional)", "max_results (optional)"]
                },
                {
                    "name": "summarize_document",
                    "description": "Summarize a specific document",
                    "parameters": ["file_path", "summary_length (optional)"]
                },
                {
                    "name": "get_document_stats",
                    "description": "Get repository statistics",
                    "parameters": []
                }
            ]
            
            print("✅ Direct mode enabled!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set up direct mode: {e}")
            return False
    
    async def discover_capabilities(self):
        """Discover available tools and resources from the server"""
        try:
            # Discover tools
            tools_result = await self.session.list_tools()
            self.available_tools = []
            
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                }
                self.available_tools.append(tool_info)
            
            # Discover resources
            try:
                resources_result = await self.session.list_resources()
                self.available_resources = []
                
                for resource in resources_result.resources:
                    resource_info = {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType if hasattr(resource, 'mimeType') else None
                    }
                    self.available_resources.append(resource_info)
                    
            except Exception as e:
                print(f"⚠️ Could not discover resources: {e}")
            
            print(f"🔍 Discovered {len(self.available_tools)} tools and {len(self.available_resources)} resources")
            
        except Exception as e:
            print(f"⚠️ Error discovering capabilities: {e}")
    
    def display_capabilities(self):
        """Display available tools and resources"""
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
        
        if self.available_resources:
            print(f"\n📚 Available Resources: {len(self.available_resources)}")
            for resource in self.available_resources[:5]:  # Show first 5
                print(f"   • {resource['name']}")
            if len(self.available_resources) > 5:
                print(f"   ... and {len(self.available_resources) - 5} more")
    
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
            if tool_name == "list_files":
                return self.doc_module.list_files(
                    kwargs.get('file_type', ''),
                    kwargs.get('include_subdirs', True)
                )
            elif tool_name == "search_documents":
                return self.doc_module.search_documents(
                    kwargs.get('query', ''),
                    kwargs.get('file_types'),
                    kwargs.get('case_sensitive', False),
                    kwargs.get('max_results', 50)
                )
            elif tool_name == "summarize_document":
                return self.doc_module.summarize_document(
                    kwargs.get('file_path', ''),
                    kwargs.get('summary_length', 'brief')
                )
            elif tool_name == "get_document_stats":
                return self.doc_module.get_document_stats()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error calling tool: {e}"
    
    async def test_file_listing(self):
        """Test file listing functionality"""
        print("\n📁 File Listing Test")
        print("-" * 30)
        
        file_type = input("Enter file type to filter (txt, md, json, csv, or leave empty): ").strip()
        include_subdirs_input = input("Include subdirectories? [y/N]: ").strip().lower()
        include_subdirs = include_subdirs_input in ['y', 'yes']
        
        print(f"\n🔄 Listing files...")
        if file_type:
            print(f"   Filter: {file_type}")
        print(f"   Include subdirs: {include_subdirs}")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("list_files", {
                "file_type": file_type,
                "include_subdirs": include_subdirs
            })
        else:
            result = self.call_tool_direct("list_files", 
                                         file_type=file_type,
                                         include_subdirs=include_subdirs)
        
        print("\n📊 Result:")
        try:
            data = json.loads(result)
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                print(f"Total files: {data.get('total_files', 0)}")
                print(f"Filter applied: {data.get('file_type_filter', 'none')}")
                
                for file_info in data.get('files', [])[:10]:  # Show first 10
                    size = file_info.get('size_human', 'Unknown')
                    path = file_info.get('relative_path', 'Unknown')
                    print(f"  📄 {path} ({size})")
                
                if len(data.get('files', [])) > 10:
                    print(f"  ... and {len(data['files']) - 10} more files")
        except:
            print(result)
    
    async def test_document_search(self):
        """Test document search functionality"""
        print("\n🔍 Document Search Test")
        print("-" * 30)
        
        query = input("Enter search query: ").strip()
        if not query:
            print("❌ Search query is required")
            return
        
        file_types_input = input("Enter file types (comma-separated, e.g., txt,md): ").strip()
        case_sensitive_input = input("Case sensitive search? [y/N]: ").strip().lower()
        max_results_input = input("Maximum results [10]: ").strip() or "10"
        
        file_types = None
        if file_types_input:
            file_types = [t.strip() for t in file_types_input.split(',')]
        
        case_sensitive = case_sensitive_input in ['y', 'yes']
        
        try:
            max_results = int(max_results_input)
        except ValueError:
            max_results = 10
        
        print(f"\n🔄 Searching for '{query}'...")
        if file_types:
            print(f"   File types: {', '.join(file_types)}")
        print(f"   Case sensitive: {case_sensitive}")
        print(f"   Max results: {max_results}")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("search_documents", {
                "query": query,
                "file_types": file_types,
                "case_sensitive": case_sensitive,
                "max_results": max_results
            })
        else:
            result = self.call_tool_direct("search_documents",
                                         query=query,
                                         file_types=file_types,
                                         case_sensitive=case_sensitive,
                                         max_results=max_results)
        
        print("\n📊 Search Results:")
        try:
            data = json.loads(result)
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                files_searched = data.get('files_searched', 0)
                matches = data.get('total_files_with_matches', 0)
                print(f"Files searched: {files_searched}")
                print(f"Files with matches: {matches}")
                
                for result_item in data.get('results', []):
                    file_path = result_item.get('file', 'Unknown')
                    match_count = result_item.get('matches', 0)
                    file_size = result_item.get('file_size', 'Unknown')
                    
                    print(f"\n📄 {file_path} ({file_size}) - {match_count} matches")
                    
                    for context in result_item.get('contexts', [])[:2]:  # Show first 2 contexts
                        line_num = context.get('line_number', 'Unknown')
                        highlight = context.get('highlight', '')
                        print(f"   Line {line_num}: {highlight[:100]}...")
        except:
            print(result)
    
    async def test_document_summary(self):
        """Test document summarization"""
        print("\n📋 Document Summarization Test")
        print("-" * 30)
        
        # First, list available files
        if hasattr(self, 'session') and self.session:
            files_result = await self.call_tool_mcp("list_files", {})
        else:
            files_result = self.call_tool_direct("list_files")
        
        try:
            files_data = json.loads(files_result)
            available_files = files_data.get('files', [])
            
            if not available_files:
                print("❌ No files available for summarization")
                return
            
            print("📁 Available files:")
            for i, file_info in enumerate(available_files[:10], 1):
                path = file_info.get('relative_path', 'Unknown')
                size = file_info.get('size_human', 'Unknown')
                print(f"  {i}. {path} ({size})")
            
            file_input = input(f"\nEnter file number (1-{min(len(available_files), 10)}) or file path: ").strip()
            
            # Parse input
            if file_input.isdigit():
                file_idx = int(file_input) - 1
                if 0 <= file_idx < len(available_files):
                    file_path = available_files[file_idx]['relative_path']
                else:
                    print("❌ Invalid file number")
                    return
            else:
                file_path = file_input
            
        except:
            file_path = input("Enter file path to summarize: ").strip()
        
        if not file_path:
            print("❌ File path is required")
            return
        
        summary_length = input("Summary length (brief/detailed) [brief]: ").strip() or "brief"
        
        print(f"\n🔄 Summarizing '{file_path}'...")
        print(f"   Length: {summary_length}")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("summarize_document", {
                "file_path": file_path,
                "summary_length": summary_length
            })
        else:
            result = self.call_tool_direct("summarize_document",
                                         file_path=file_path,
                                         summary_length=summary_length)
        
        print("\n📊 Summary Result:")
        try:
            data = json.loads(result)
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                # Display file info
                metadata = data.get('file_metadata', {})
                print(f"📄 File: {data.get('file_path', 'Unknown')}")
                print(f"   Size: {metadata.get('size_human', 'Unknown')}")
                
                # Display content analysis
                analysis = data.get('content_analysis', {})
                print(f"\n📊 Content Analysis:")
                print(f"   Words: {analysis.get('word_count', 0)}")
                print(f"   Lines: {analysis.get('total_lines', 0)}")
                print(f"   Reading time: {analysis.get('estimated_reading_time_minutes', 0)} minutes")
                
                # Display summary
                summary = data.get('summary', '')
                if summary:
                    print(f"\n📝 Summary ({data.get('summary_type', 'brief')}):")
                    print(f"   {summary}")
                
                # Display keywords
                keywords = data.get('top_keywords', [])
                if keywords:
                    print(f"\n🏷️ Top Keywords: {', '.join(keywords[:5])}")
        except:
            print(result)
    
    async def test_document_stats(self):
        """Test document repository statistics"""
        print("\n📊 Document Repository Statistics")
        print("-" * 30)
        
        print("🔄 Gathering repository statistics...")
        
        if hasattr(self, 'session') and self.session:
            result = await self.call_tool_mcp("get_document_stats", {})
        else:
            result = self.call_tool_direct("get_document_stats")
        
        print("\n📊 Repository Stats:")
        try:
            data = json.loads(result)
            if "error" in data:
                print(f"❌ Error: {data['error']}")
            else:
                # Basic stats
                print(f"📁 Repository: {data.get('repository_path', 'Unknown')}")
                print(f"📄 Total files: {data.get('total_files', 0)}")
                
                total_size = data.get('total_size', {})
                print(f"💾 Total size: {total_size.get('human', 'Unknown')}")
                
                # Supported extensions
                extensions = data.get('supported_extensions', [])
                print(f"🔧 Supported types: {', '.join(sorted(extensions))}")
                
                # By file type
                by_type = data.get('by_file_type', {})
                if by_type:
                    print(f"\n📊 Files by type:")
                    for ext, info in by_type.items():
                        count = info.get('count', 0)
                        size = info.get('size_human', 'Unknown')
                        print(f"   {ext}: {count} files ({size})")
                
                # Largest files
                largest = data.get('largest_files', [])
                if largest:
                    print(f"\n📈 Largest files:")
                    for file_info in largest[:3]:
                        name = file_info.get('relative_path', 'Unknown')
                        size = file_info.get('size_human', 'Unknown')
                        print(f"   📄 {name} ({size})")
        except:
            print(result)
    
    async def test_resource_access(self):
        """Test reading documents as resources"""
        print("\n📚 Resource Access Test")
        print("-" * 30)
        
        if not hasattr(self, 'session') or not self.session:
            print("⚠️ Resource access only available in MCP mode")
            return
        
        try:
            # List available resources
            resources_result = await self.session.list_resources()
            resources = resources_result.resources
            
            if not resources:
                print("❌ No resources available")
                return
            
            print("📚 Available resources:")
            for i, resource in enumerate(resources[:10], 1):
                print(f"  {i}. {resource.name}")
            
            if len(resources) > 10:
                print(f"  ... and {len(resources) - 10} more")
            
            resource_input = input(f"\nEnter resource number (1-{min(len(resources), 10)}): ").strip()
            
            if not resource_input.isdigit():
                print("❌ Invalid resource number")
                return
            
            resource_idx = int(resource_input) - 1
            if not (0 <= resource_idx < len(resources)):
                print("❌ Invalid resource number")
                return
            
            selected_resource = resources[resource_idx]
            print(f"\n🔄 Reading resource: {selected_resource.name}")
            
            # Read the resource
            result = await self.session.read_resource(selected_resource.uri)
            content = result.contents[0].text
            
            print(f"\n📄 Content Preview (first 500 characters):")
            print("-" * 50)
            print(content[:500])
            if len(content) > 500:
                print("\n... (truncated)")
            print("-" * 50)
            print(f"Total content length: {len(content)} characters")
            
        except Exception as e:
            print(f"❌ Error accessing resource: {e}")
    
    async def run_interactive_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "=" * 60)
            print("📄 Interactive Document Processing Tester")
            print("=" * 60)
            print("1. 📁 Test File Listing")
            print("2. 🔍 Test Document Search")
            print("3. 📋 Test Document Summarization")
            print("4. 📊 Get Repository Statistics")
            print("5. 📚 Test Resource Access (MCP mode only)")
            print("6. 🔧 Show Available Tools & Resources")
            print("7. 📂 Create Sample Documents")
            print("8. 🚪 Exit")
            
            choice = input("\nSelect an option (1-8): ").strip()
            
            try:
                if choice == "1":
                    await self.test_file_listing()
                elif choice == "2":
                    await self.test_document_search()
                elif choice == "3":
                    await self.test_document_summary()
                elif choice == "4":
                    await self.test_document_stats()
                elif choice == "5":
                    await self.test_resource_access()
                elif choice == "6":
                    self.display_capabilities()
                elif choice == "7":
                    await self.create_sample_documents()
                elif choice == "8":
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-8.")
            
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation cancelled")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            if choice != "8":
                input("\nPress Enter to continue...")
    
    async def create_sample_documents(self):
        """Create or recreate sample documents"""
        print("\n📂 Create Sample Documents")
        print("-" * 30)
        
        if not self.test_directory:
            new_dir = input("Enter directory path for sample documents: ").strip()
            if new_dir:
                self.test_directory = new_dir
            else:
                print("❌ Directory path required")
                return
        
        print(f"📁 Creating sample documents in: {self.test_directory}")
        success = await self.setup_test_documents(self.test_directory)
        
        if success:
            print("✅ Sample documents created successfully!")
            print("You can now test the document processing features.")
        else:
            print("❌ Failed to create sample documents")
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Document Processing Tester")
    parser.add_argument("directory", nargs="?", help="Directory containing documents to process")
    parser.add_argument("--create-samples", action="store_true", help="Create sample documents")
    
    args = parser.parse_args()
    
    # Determine test directory
    test_directory = args.directory
    
    if not test_directory:
        if args.create_samples:
            test_directory = input("Enter directory path for sample documents: ").strip()
        else:
            test_directory = input("Enter directory path containing documents: ").strip()
    
    if not test_directory:
        print("❌ Directory path is required")
        sys.exit(1)
    
    tester = InteractiveDocumentTester(test_directory)
    
    try:
        print("📄 Starting Interactive Document Processing Tester")
        print("=" * 60)
        
        # Create sample documents if requested
        if args.create_samples or not Path(test_directory).exists():
            print(f"📂 Setting up test documents in: {test_directory}")
            if not await tester.setup_test_documents(test_directory):
                print("❌ Failed to create test documents")
                return
        
        # Connect to server
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