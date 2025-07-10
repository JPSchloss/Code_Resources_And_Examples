# document_processing_server.py
import os
import json
import mimetypes
import asyncio
import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF processing disabled")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available - DOCX processing disabled")

# Import MCP components
from mcp.server.fastmcp import FastMCP

class DocumentProcessingHandler:
    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory).resolve()
        self.allowed_extensions = {'.txt', '.md', '.json', '.csv'}
        
        # Add PDF and DOCX if libraries are available
        if PDF_AVAILABLE:
            self.allowed_extensions.add('.pdf')
        if DOCX_AVAILABLE:
            self.allowed_extensions.add('.docx')
        
        # Ensure root directory exists
        if not self.root_directory.exists():
            raise ValueError(f"Root directory does not exist: {root_directory}")
        
        # Security: ensure it's a directory
        if not self.root_directory.is_dir():
            raise ValueError(f"Root path is not a directory: {root_directory}")
        
        # Maximum file size (10MB default)
        self.max_file_size = 10 * 1024 * 1024
        
        logger.info(f"Document processor initialized for: {self.root_directory}")
        logger.info(f"Supported extensions: {', '.join(sorted(self.allowed_extensions))}")
    
    def _is_allowed_file(self, file_path: Path) -> bool:
        """Check if file type is allowed for processing"""
        return file_path.suffix.lower() in self.allowed_extensions
    
    def _is_safe_path(self, file_path: Path) -> bool:
        """Security check to prevent path traversal attacks"""
        try:
            resolved_path = file_path.resolve()
            resolved_path.relative_to(self.root_directory)
            return True
        except (ValueError, OSError):
            return False
    
    def _check_file_size(self, file_path: Path) -> bool:
        """Check if file size is within limits"""
        try:
            return file_path.stat().st_size <= self.max_file_size
        except OSError:
            return False
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from various file types"""
        if not self._is_safe_path(file_path):
            return f"Security error: Invalid file path"
        
        if not self._check_file_size(file_path):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return f"File too large: {size_mb:.1f}MB (max: {self.max_file_size / (1024 * 1024)}MB)"
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.txt', '.md']:
                return file_path.read_text(encoding='utf-8', errors='ignore')
            
            elif suffix == '.pdf' and PDF_AVAILABLE:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page.extract_text() + "\n"
                        except Exception as e:
                            text += f"\n[Error reading page {page_num + 1}: {e}]\n"
                return text
            
            elif suffix == '.docx' and DOCX_AVAILABLE:
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            elif suffix == '.json':
                data = json.loads(file_path.read_text(encoding='utf-8'))
                return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif suffix == '.csv':
                return file_path.read_text(encoding='utf-8', errors='ignore')
            
            else:
                return f"Unsupported file type: {suffix}"
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata about a file"""
        try:
            stat = file_path.stat()
            return {
                "size_bytes": stat.st_size,
                "size_human": self._format_file_size(stat.st_size),
                "modified_time": stat.st_mtime,
                "extension": file_path.suffix.lower(),
                "name": file_path.name,
                "relative_path": str(file_path.relative_to(self.root_directory))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def discover_files(self) -> List[Dict[str, Any]]:
        """Discover all processable files in the directory tree"""
        files = []
        try:
            for file_path in self.root_directory.rglob('*'):
                if (file_path.is_file() and 
                    self._is_allowed_file(file_path) and 
                    self._is_safe_path(file_path)):
                    
                    metadata = self.get_file_metadata(file_path)
                    if "error" not in metadata:
                        files.append(metadata)
        except Exception as e:
            logger.error(f"Error discovering files: {e}")
        
        return sorted(files, key=lambda x: x.get('relative_path', ''))

# Initialize the handler (will be set by command line argument)
doc_handler = None

# Create the MCP server instance
mcp = FastMCP("document-processor")

def _run_async_in_sync(coro):
    """Helper function to run async code in sync context"""
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

@mcp.resource("file://{path}")
def read_document(path: str) -> str:
    """Read content from a specific document
    
    Args:
        path: Relative path to the document
    """
    if doc_handler is None:
        return "Error: Document handler not initialized"
    
    try:
        file_path = doc_handler.root_directory / path
        
        if not file_path.exists():
            return f"File not found: {path}"
        
        if not doc_handler._is_allowed_file(file_path):
            return f"File type not allowed: {file_path.suffix}"
        
        return doc_handler._extract_text_from_file(file_path)
        
    except Exception as e:
        logger.error(f"Error reading document {path}: {e}")
        return f"Error reading document: {str(e)}"

@mcp.tool()
def list_files(file_type: str = "", include_subdirs: bool = True) -> str:
    """List all files of a specific type in the repository
    
    Args:
        file_type: File extension to filter by (e.g., 'pdf', 'txt', leave empty for all)
        include_subdirs: Whether to include files in subdirectories
    """
    if doc_handler is None:
        return json.dumps({"error": "Document handler not initialized"}, indent=2)
    
    try:
        files = doc_handler.discover_files()
        
        # Filter by file type if specified
        if file_type:
            file_ext = f".{file_type.lower().lstrip('.')}"
            files = [f for f in files if f.get('extension') == file_ext]
        
        # Filter by subdirectories if specified
        if not include_subdirs:
            files = [f for f in files if '/' not in f.get('relative_path', '')]
        
        result = {
            "total_files": len(files),
            "file_type_filter": file_type or "all",
            "include_subdirs": include_subdirs,
            "files": files
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return json.dumps({"error": f"Error listing files: {str(e)}"}, indent=2)

@mcp.tool()
def search_documents(query: str, file_types: List[str] = None, case_sensitive: bool = False, max_results: int = 50) -> str:
    """Search for text across all documents
    
    Args:
        query: Search term to find in documents
        file_types: List of file types to search (e.g., ['txt', 'pdf'])
        case_sensitive: Whether search should be case sensitive
        max_results: Maximum number of results to return
    """
    if doc_handler is None:
        return json.dumps({"error": "Document handler not initialized"}, indent=2)
    
    if not query.strip():
        return json.dumps({"error": "Search query cannot be empty"}, indent=2)
    
    if file_types is None:
        file_types = ["txt", "pdf", "docx", "md", "json", "csv"]
    
    async def _search_documents():
        search_query = query if case_sensitive else query.lower()
        results = []
        files_searched = 0
        
        try:
            for file_path in doc_handler.root_directory.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower().lstrip('.') in file_types and
                    doc_handler._is_allowed_file(file_path) and
                    doc_handler._is_safe_path(file_path)):
                    
                    files_searched += 1
                    
                    try:
                        content = doc_handler._extract_text_from_file(file_path)
                        if content.startswith("Error") or content.startswith("Security error"):
                            continue
                        
                        search_content = content if case_sensitive else content.lower()
                        
                        if search_query in search_content:
                            # Find context around matches
                            lines = content.split('\n')
                            matching_lines = []
                            
                            for i, line in enumerate(lines):
                                line_to_search = line if case_sensitive else line.lower()
                                if search_query in line_to_search:
                                    # Include context (previous and next lines)
                                    start = max(0, i - 1)
                                    end = min(len(lines), i + 2)
                                    context = '\n'.join(lines[start:end])
                                    
                                    matching_lines.append({
                                        "line_number": i + 1,
                                        "context": context.strip(),
                                        "highlight": line.strip()
                                    })
                            
                            if matching_lines:
                                metadata = doc_handler.get_file_metadata(file_path)
                                results.append({
                                    "file": str(file_path.relative_to(doc_handler.root_directory)),
                                    "file_size": metadata.get("size_human", "Unknown"),
                                    "matches": len(matching_lines),
                                    "contexts": matching_lines[:5]  # Limit to first 5 matches per file
                                })
                                
                                # Stop if we've reached max results
                                if len(results) >= max_results:
                                    break
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                "error": f"Search error: {str(e)}",
                "query": query,
                "files_searched": files_searched
            }
        
        return {
            "query": query,
            "search_options": {
                "case_sensitive": case_sensitive,
                "file_types": file_types,
                "max_results": max_results
            },
            "files_searched": files_searched,
            "total_files_with_matches": len(results),
            "results": results
        }
    
    try:
        result = _run_async_in_sync(_search_documents())
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Search tool execution error: {e}")
        return json.dumps({"error": f"Error executing search: {str(e)}"}, indent=2)

@mcp.tool()
def summarize_document(file_path: str, summary_length: str = "brief") -> str:
    """Extract key information and summarize a specific document
    
    Args:
        file_path: Relative path to the document to summarize
        summary_length: Length of summary ('brief' or 'detailed')
    """
    if doc_handler is None:
        return json.dumps({"error": "Document handler not initialized"}, indent=2)
    
    try:
        full_path = doc_handler.root_directory / file_path
        
        # Security check
        if not doc_handler._is_safe_path(full_path):
            return json.dumps({"error": "Invalid file path"}, indent=2)
        
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {file_path}"}, indent=2)
        
        content = doc_handler._extract_text_from_file(full_path)
        
        if content.startswith("Error") or content.startswith("Security error"):
            return json.dumps({"error": content}, indent=2)
        
        # Extract document metadata
        metadata = doc_handler.get_file_metadata(full_path)
        
        # Analyze content
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        words = content.split()
        
        # Simple sentence extraction for summarization
        sentences = []
        for line in non_empty_lines:
            line_sentences = line.replace('\n', ' ').split('.')
            sentences.extend([s.strip() for s in line_sentences if s.strip() and len(s.strip()) > 10])
        
        # Select sentences based on summary length
        if summary_length == "brief":
            summary_sentences = sentences[:3]
        else:  # detailed
            summary_sentences = sentences[:7]
        
        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        # Extract keywords (simple frequency analysis)
        word_freq = {}
        for word in words:
            word_clean = word.lower().strip('.,!?";()[]{}')
            if len(word_clean) > 3 and word_clean.isalpha():
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary_info = {
            "file_path": file_path,
            "file_metadata": metadata,
            "content_analysis": {
                "total_lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "estimated_reading_time_minutes": round(len(words) / 200, 1)
            },
            "summary": summary,
            "summary_type": summary_length,
            "top_keywords": [keyword for keyword, freq in top_keywords],
            "processing_timestamp": time.time()
        }
        
        return json.dumps(summary_info, indent=2)
        
    except Exception as e:
        logger.error(f"Error summarizing document {file_path}: {e}")
        return json.dumps({"error": f"Error processing file: {str(e)}"}, indent=2)

@mcp.tool()
def get_document_stats() -> str:
    """Get statistics about all documents in the repository"""
    if doc_handler is None:
        return json.dumps({"error": "Document handler not initialized"}, indent=2)
    
    try:
        files = doc_handler.discover_files()
        
        # Calculate statistics
        total_files = len(files)
        total_size = sum(f.get('size_bytes', 0) for f in files)
        
        # Group by file type
        by_type = {}
        for file_info in files:
            ext = file_info.get('extension', 'unknown')
            if ext not in by_type:
                by_type[ext] = {'count': 0, 'size_bytes': 0}
            by_type[ext]['count'] += 1
            by_type[ext]['size_bytes'] += file_info.get('size_bytes', 0)
        
        # Format sizes
        for ext_info in by_type.values():
            ext_info['size_human'] = doc_handler._format_file_size(ext_info['size_bytes'])
        
        stats = {
            "repository_path": str(doc_handler.root_directory),
            "total_files": total_files,
            "total_size": {
                "bytes": total_size,
                "human": doc_handler._format_file_size(total_size)
            },
            "supported_extensions": list(doc_handler.allowed_extensions),
            "by_file_type": by_type,
            "largest_files": sorted(files, key=lambda x: x.get('size_bytes', 0), reverse=True)[:5]
        }
        
        return json.dumps(stats, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return json.dumps({"error": f"Error getting statistics: {str(e)}"}, indent=2)

# Server startup
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python document_processing_server.py <root_directory>")
        print("Example: python document_processing_server.py ./documents")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    try:
        # Initialize the global handler
        doc_handler = DocumentProcessingHandler(root_dir)
        
        logger.info("Starting Document Processing MCP Server...")
        logger.info("Press Ctrl+C to stop the server")
        
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise