# storage/embedding_rag_exporter.py

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class EmbeddingRAGExporter:
    """Enhanced RAG exporter optimized for dual LLM systems."""

    def __init__(self):
        self.max_chunk_size = 1000  # Optimal for most embedding models
        self.overlap_size = 100  # For context preservation
        self.min_chunk_size = 50  # Minimum useful chunk size

    def export_for_dual_llm(self, snippets_data: List[Dict],
                            output_dir: str,
                            query: str) -> Dict[str, str]:
        """Export in format optimized for chat LLM + code LLM workflow."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = self._sanitize_filename(query)

        exports = {}

        # 1. Chat LLM optimized format (natural language descriptions)
        chat_file = self._export_for_chat_llm(snippets_data, output_path, safe_query, timestamp)
        exports['chat_llm'] = chat_file

        # 2. Code LLM optimized format (pure code with minimal context)
        code_file = self._export_for_code_llm(snippets_data, output_path, safe_query, timestamp)
        exports['code_llm'] = code_file

        # 3. Unified format for cross-referencing
        unified_file = self._export_unified_format(snippets_data, output_path, safe_query, timestamp)
        exports['unified'] = unified_file

        # 4. Vector database ready format
        vector_file = self._export_for_vector_db(snippets_data, output_path, safe_query, timestamp)
        exports['vector_db'] = vector_file

        # 5. Embedding-ready chunks
        chunks_file = self._export_embedding_chunks(snippets_data, output_path, safe_query, timestamp)
        exports['embedding_chunks'] = chunks_file

        return exports

    def _export_for_chat_llm(self, snippets_data: List[Dict], output_path: Path,
                             query: str, timestamp: str) -> str:
        """Export format optimized for chat LLM understanding."""
        filename = output_path / f"chat_llm_{query}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            for i, snippet_data in enumerate(snippets_data):
                # Create natural language description
                description = self._generate_natural_description(snippet_data)

                doc = {
                    'id': f"chat_{query}_{i}",
                    'type': 'code_explanation',
                    'content': description,
                    'code_summary': self._extract_code_summary(snippet_data['code']),
                    'use_cases': self._generate_use_cases(snippet_data),
                    'difficulty': snippet_data['metadata'].get('complexity', 'intermediate'),
                    'categories': snippet_data['metadata'].get('categories', []),
                    'patterns': snippet_data['metadata'].get('patterns', []),
                    'freelance_value': snippet_data['metadata'].get('client_value', 'general_purpose'),
                    'implementation_tips': self._generate_implementation_tips(snippet_data),
                    'common_pitfalls': self._identify_common_pitfalls(snippet_data),
                    'related_concepts': self._suggest_related_concepts(snippet_data),
                    'original_query': query,
                    'timestamp': timestamp
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        return str(filename)

    def _export_for_code_llm(self, snippets_data: List[Dict], output_path: Path,
                             query: str, timestamp: str) -> str:
        """Export format optimized for code LLM."""
        filename = output_path / f"code_llm_{query}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            for i, snippet_data in enumerate(snippets_data):
                # Chunk large code snippets
                chunks = self._chunk_code(snippet_data['code'])

                for j, chunk in enumerate(chunks):
                    doc = {
                        'id': f"code_{query}_{i}_{j}",
                        'type': 'code_implementation',
                        'code': chunk,
                        'language': 'python',
                        'patterns': snippet_data['metadata'].get('patterns', []),
                        'imports': snippet_data['metadata'].get('imports', []),
                        'complexity': snippet_data['metadata'].get('complexity', 'intermediate'),
                        'functions': self._extract_function_signatures(chunk),
                        'classes': self._extract_class_names(chunk),
                        'dependencies': snippet_data['metadata'].get('imports', []),
                        'code_style': self._analyze_code_style(chunk),
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'original_query': query,
                        'timestamp': timestamp
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        return str(filename)

    def _export_unified_format(self, snippets_data: List[Dict], output_path: Path,
                               query: str, timestamp: str) -> str:
        """Export unified format for cross-referencing between LLMs."""
        filename = output_path / f"unified_{query}_{timestamp}.json"

        unified_data = {
            'query': query,
            'timestamp': timestamp,
            'total_snippets': len(snippets_data),
            'snippet_index': {},
            'category_index': {},
            'pattern_index': {},
            'complexity_distribution': {},
            'freelance_value_distribution': {},
            'import_frequency': {},
            'query_suggestions': []
        }

        # Build indices for fast lookup
        import_counter = {}
        freelance_values = {}

        for i, snippet_data in enumerate(snippets_data):
            snippet_id = f"{query}_{i}"

            # Snippet index
            unified_data['snippet_index'][snippet_id] = {
                'chat_llm_id': f"chat_{snippet_id}",
                'code_llm_ids': [f"code_{snippet_id}_{j}" for j in range(len(self._chunk_code(snippet_data['code'])))],
                'score': snippet_data.get('score', 0),
                'source': snippet_data['metadata'].get('source', ''),
                'hash': hashlib.md5(snippet_data['code'].encode()).hexdigest(),
                'size_bytes': len(snippet_data['code'].encode()),
                'lines_count': len(snippet_data['code'].splitlines())
            }

            # Category index
            for category in snippet_data['metadata'].get('categories', []):
                if category not in unified_data['category_index']:
                    unified_data['category_index'][category] = []
                unified_data['category_index'][category].append(snippet_id)

            # Pattern index
            for pattern in snippet_data['metadata'].get('patterns', []):
                if pattern not in unified_data['pattern_index']:
                    unified_data['pattern_index'][pattern] = []
                unified_data['pattern_index'][pattern].append(snippet_id)

            # Complexity distribution
            complexity = snippet_data['metadata'].get('complexity', 'intermediate')
            unified_data['complexity_distribution'][complexity] = unified_data['complexity_distribution'].get(
                complexity, 0) + 1

            # Freelance value distribution
            freelance_value = snippet_data['metadata'].get('client_value', 'general_purpose')
            freelance_values[freelance_value] = freelance_values.get(freelance_value, 0) + 1

            # Import frequency
            for imp in snippet_data['metadata'].get('imports', []):
                import_counter[imp] = import_counter.get(imp, 0) + 1

        unified_data['freelance_value_distribution'] = freelance_values
        unified_data['import_frequency'] = dict(sorted(import_counter.items(), key=lambda x: x[1], reverse=True)[:20])

        # Generate query suggestions based on content
        unified_data['query_suggestions'] = self._generate_query_suggestions(snippets_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)

        return str(filename)

    def _export_for_vector_db(self, snippets_data: List[Dict], output_path: Path,
                              query: str, timestamp: str) -> str:
        """Export format ready for vector database ingestion."""
        filename = output_path / f"vector_db_{query}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            for i, snippet_data in enumerate(snippets_data):
                # Combine code and description for better embeddings
                combined_text = self._create_embedding_text(snippet_data)

                doc = {
                    'id': f"vector_{query}_{i}",
                    'text': combined_text,
                    'metadata': {
                        'type': 'code_snippet',
                        'language': 'python',
                        'query': query,
                        'score': snippet_data.get('score', 0),
                        'complexity': snippet_data['metadata'].get('complexity', 'intermediate'),
                        'categories': snippet_data['metadata'].get('categories', []),
                        'patterns': snippet_data['metadata'].get('patterns', []),
                        'imports': snippet_data['metadata'].get('imports', []),
                        'source': snippet_data['metadata'].get('source', ''),
                        'lines': snippet_data['metadata'].get('lines', 0),
                        'has_docstring': snippet_data['metadata'].get('has_docstring', False),
                        'freelance_score': snippet_data['metadata'].get('freelance_score', 0),
                        'client_value': snippet_data['metadata'].get('client_value', 'general_purpose'),
                        'use_cases': snippet_data['metadata'].get('use_cases', []),
                        'timestamp': timestamp
                    }
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        return str(filename)

    def _export_embedding_chunks(self, snippets_data: List[Dict], output_path: Path,
                                 query: str, timestamp: str) -> str:
        """Export pre-chunked format optimized for embeddings."""
        filename = output_path / f"embedding_chunks_{query}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            chunk_id = 0

            for i, snippet_data in enumerate(snippets_data):
                # Create multiple types of chunks for comprehensive coverage

                # 1. Code-only chunks
                code_chunks = self._chunk_code(snippet_data['code'])
                for j, chunk in enumerate(code_chunks):
                    if len(chunk.strip()) >= self.min_chunk_size:
                        doc = {
                            'id': f"chunk_{query}_{chunk_id}",
                            'parent_id': f"snippet_{query}_{i}",
                            'chunk_type': 'code_only',
                            'text': chunk,
                            'chunk_index': j,
                            'total_chunks': len(code_chunks),
                            'metadata': {
                                'source_snippet_id': i,
                                'complexity': snippet_data['metadata'].get('complexity', 'intermediate'),
                                'categories': snippet_data['metadata'].get('categories', [])[:3],
                                # Limit for efficiency
                                'patterns': snippet_data['metadata'].get('patterns', [])[:3],
                                'query': query,
                                'timestamp': timestamp
                            }
                        }
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        chunk_id += 1

                # 2. Description-only chunk
                description = self._generate_natural_description(snippet_data)
                if len(description) >= self.min_chunk_size:
                    doc = {
                        'id': f"chunk_{query}_{chunk_id}",
                        'parent_id': f"snippet_{query}_{i}",
                        'chunk_type': 'description_only',
                        'text': description,
                        'metadata': {
                            'source_snippet_id': i,
                            'complexity': snippet_data['metadata'].get('complexity', 'intermediate'),
                            'categories': snippet_data['metadata'].get('categories', [])[:3],
                            'use_cases': snippet_data['metadata'].get('use_cases', [])[:3],
                            'query': query,
                            'timestamp': timestamp
                        }
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    chunk_id += 1

                # 3. Mixed chunks (code + context)
                mixed_text = self._create_embedding_text(snippet_data)
                mixed_chunks = self._chunk_text(mixed_text)
                for j, chunk in enumerate(mixed_chunks):
                    if len(chunk.strip()) >= self.min_chunk_size:
                        doc = {
                            'id': f"chunk_{query}_{chunk_id}",
                            'parent_id': f"snippet_{query}_{i}",
                            'chunk_type': 'mixed',
                            'text': chunk,
                            'chunk_index': j,
                            'total_chunks': len(mixed_chunks),
                            'metadata': {
                                'source_snippet_id': i,
                                'complexity': snippet_data['metadata'].get('complexity', 'intermediate'),
                                'categories': snippet_data['metadata'].get('categories', [])[:3],
                                'patterns': snippet_data['metadata'].get('patterns', [])[:3],
                                'imports': snippet_data['metadata'].get('imports', [])[:5],
                                'query': query,
                                'timestamp': timestamp
                            }
                        }
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        chunk_id += 1

        return str(filename)

    def _generate_natural_description(self, snippet_data: Dict) -> str:
        """Generate natural language description for chat LLM."""
        code = snippet_data['code']
        metadata = snippet_data['metadata']

        description_parts = []

        # Start with what the code does
        if 'class' in code.lower():
            description_parts.append("This is a Python class implementation")
        elif 'def ' in code:
            description_parts.append("This is a Python function implementation")
        elif 'async def' in code:
            description_parts.append("This is an asynchronous Python function")
        else:
            description_parts.append("This is a Python code snippet")

        # Add complexity and purpose
        complexity = metadata.get('complexity', 'intermediate')
        description_parts.append(f"at {complexity} difficulty level")

        # Add categories
        categories = metadata.get('categories', [])
        if categories:
            description_parts.append(f"related to {', '.join(categories[:3])}")

        # Add patterns
        patterns = metadata.get('patterns', [])
        if patterns:
            description_parts.append(f"using patterns: {', '.join(patterns[:3])}")

        # Add imports context
        imports = metadata.get('imports', [])
        if imports:
            description_parts.append(f"utilizing libraries: {', '.join(imports[:3])}")

        # Add freelance context
        client_value = metadata.get('client_value', '')
        if 'high_value' in client_value:
            description_parts.append("with high client value for freelance projects")

        description = ". ".join(description_parts) + "."

        # Add use cases
        use_cases = metadata.get('use_cases', [])
        if use_cases:
            description += f" This code is useful for: {', '.join(use_cases[:3])}."

        # Add the actual code with explanation
        description += f"\n\nCode implementation:\n```python\n{code}\n```"

        return description

    def _extract_code_summary(self, code: str) -> str:
        """Extract a brief summary of what the code does."""
        lines = code.strip().split('\n')

        # Look for docstrings
        for line in lines:
            if '"""' in line or "'''" in line:
                # Extract docstring content
                docstring_match = re.search(r'["\']([^"\']+)["\']', line)
                if docstring_match:
                    return docstring_match.group(1)

        # Look for comments
        for line in lines:
            if line.strip().startswith('#'):
                return line.strip()[1:].strip()

        # Analyze function/class names
        if 'def ' in code:
            func_matches = re.findall(r'def\s+(\w+)', code)
            if func_matches:
                return f"Implements {', '.join(func_matches[:3])} function(s)"

        if 'class ' in code:
            class_matches = re.findall(r'class\s+(\w+)', code)
            if class_matches:
                return f"Defines {', '.join(class_matches[:2])} class(es)"

        # Fallback to first meaningful line
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('import', 'from', '#')):
                return f"Code starting with: {stripped[:50]}..."

        return "Python code implementation"

    def _generate_use_cases(self, snippet_data: Dict) -> List[str]:
        """Generate practical use cases for freelance work."""
        categories = snippet_data['metadata'].get('categories', [])
        patterns = snippet_data['metadata'].get('patterns', [])
        imports = snippet_data['metadata'].get('imports', [])

        use_cases = []

        # Category-based use cases
        if 'api_development' in categories:
            use_cases.extend(["Building REST APIs for client applications", "Creating microservices architecture"])
        if 'data_processing' in categories:
            use_cases.extend(["Data analysis and processing scripts", "ETL pipeline development"])
        if 'web_scraping' in categories:
            use_cases.extend(["Automated data collection projects", "Market research automation"])
        if 'testing' in categories:
            use_cases.extend(["Test automation and quality assurance", "CI/CD pipeline testing"])
        if 'database' in categories:
            use_cases.extend(["Database integration and management", "Data migration scripts"])
        if 'cli_tools' in categories:
            use_cases.extend(["Command-line utility development", "DevOps automation tools"])
        if 'client_integrations' in categories:
            use_cases.extend(["Third-party API integrations", "Payment system implementations"])

        # Import-based use cases
        if 'fastapi' in imports:
            use_cases.append("Modern web API development")
        if 'pandas' in imports:
            use_cases.append("Data analysis and reporting")
        if 'selenium' in imports:
            use_cases.append("Web automation and testing")
        if 'stripe' in imports:
            use_cases.append("E-commerce payment processing")

        return use_cases or ["General Python development"]

    def _generate_implementation_tips(self, snippet_data: Dict) -> List[str]:
        """Generate implementation tips based on code analysis."""
        code = snippet_data['code']
        patterns = snippet_data['metadata'].get('patterns', [])
        categories = snippet_data['metadata'].get('categories', [])

        tips = []

        if 'async_function' in patterns:
            tips.append("Use asyncio.run() for top-level async execution")
            tips.append("Consider using async context managers for resource management")

        if 'api_development' in categories:
            tips.append("Always implement proper error handling and status codes")
            tips.append("Use request validation and response models")
            tips.append("Implement rate limiting for production APIs")

        if 'database' in categories:
            tips.append("Use connection pooling for better performance")
            tips.append("Implement proper transaction handling")
            tips.append("Consider using database migrations for schema changes")

        if 'testing' in categories:
            tips.append("Use fixtures to reduce test setup duplication")
            tips.append("Mock external dependencies in unit tests")
            tips.append("Implement both unit and integration tests")

        if 'exception_handling' in patterns:
            tips.append("Log exceptions with sufficient context for debugging")
            tips.append("Use specific exception types rather than broad catches")

        return tips

    def _identify_common_pitfalls(self, snippet_data: Dict) -> List[str]:
        """Identify common pitfalls based on code patterns."""
        code = snippet_data['code']
        patterns = snippet_data['metadata'].get('patterns', [])
        categories = snippet_data['metadata'].get('categories', [])

        pitfalls = []

        if 'async_function' in patterns:
            pitfalls.append("Don't mix async and sync code without proper handling")
            pitfalls.append("Avoid blocking operations in async functions")

        if 'database' in categories:
            pitfalls.append("Always close database connections properly")
            pitfalls.append("Avoid SQL injection by using parameterized queries")

        if 'api_development' in categories:
            pitfalls.append("Don't expose sensitive data in API responses")
            pitfalls.append("Validate all input data to prevent security issues")

        if 'exception_handling' in patterns:
            pitfalls.append("Don't catch exceptions silently without logging")
            pitfalls.append("Avoid catching Exception base class unless necessary")

        if 'file_operations' in categories:
            pitfalls.append("Always use context managers for file operations")
            pitfalls.append("Handle file not found and permission errors")

        return pitfalls

    def _suggest_related_concepts(self, snippet_data: Dict) -> List[str]:
        """Suggest related concepts to explore."""
        categories = snippet_data['metadata'].get('categories', [])
        patterns = snippet_data['metadata'].get('patterns', [])

        concepts = []

        if 'api_development' in categories:
            concepts.extend(["OpenAPI documentation", "API versioning", "Authentication middleware"])

        if 'async_programming' in categories:
            concepts.extend(["Event loops", "Coroutines", "Asyncio patterns"])

        if 'testing' in categories:
            concepts.extend(["Test-driven development", "Behavior-driven development", "Property-based testing"])

        if 'data_processing' in categories:
            concepts.extend(["Data pipelines", "Stream processing", "Data validation"])

        if 'decorators' in patterns:
            concepts.extend(["Functools", "Metaclasses", "Descriptor protocol"])

        return concepts

    def _chunk_code(self, code: str) -> List[str]:
        """Chunk large code snippets for better processing."""
        if len(code) <= self.max_chunk_size:
            return [code]

        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))

                # Start new chunk with overlap
                overlap_lines = current_chunk[-self.overlap_size // 20:] if len(
                    current_chunk) > self.overlap_size // 20 else current_chunk
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text content for embedding optimization."""
        if len(text) <= self.max_chunk_size:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            para_size = len(paragraph) + 2  # +2 for double newline

            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _create_embedding_text(self, snippet_data: Dict) -> str:
        """Create optimized text for embedding generation."""
        code = snippet_data['code']
        metadata = snippet_data['metadata']

        # Combine code with contextual information
        embedding_parts = []

        # Add summary
        summary = self._extract_code_summary(code)
        embedding_parts.append(f"Summary: {summary}")

        # Add categories as context
        categories = metadata.get('categories', [])
        if categories:
            embedding_parts.append(f"Categories: {', '.join(categories)}")

        # Add patterns as context
        patterns = metadata.get('patterns', [])
        if patterns:
            embedding_parts.append(f"Patterns: {', '.join(patterns)}")

        # Add use cases
        use_cases = metadata.get('use_cases', [])
        if use_cases:
            embedding_parts.append(f"Use cases: {', '.join(use_cases)}")

        # Add the code itself
        embedding_parts.append(f"Code:\n{code}")

        return '\n'.join(embedding_parts)

    def _extract_function_signatures(self, code: str) -> List[str]:
        """Extract function signatures from code."""
        signatures = []
        lines = code.split('\n')

        for line in lines:
            # Match function definitions
            func_match = re.match(r'\s*(async\s+)?def\s+(\w+)\s*\([^)]*\)', line)
            if func_match:
                signatures.append(line.strip())

        return signatures

    def _extract_class_names(self, code: str) -> List[str]:
        """Extract class names from code."""
        class_names = []
        class_matches = re.findall(r'class\s+(\w+)', code)
        return class_matches

    def _analyze_code_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style characteristics."""
        return {
            'has_type_hints': 'typing' in code or '->' in code or ':' in code,
            'has_docstrings': '"""' in code or "'''" in code,
            'has_comments': '#' in code,
            'uses_f_strings': re.search(r'f["\'].*{.*}.*["\']', code) is not None,
            'has_error_handling': 'try:' in code or 'except' in code,
            'async_code': 'async' in code or 'await' in code
        }

    def _generate_query_suggestions(self, snippets_data: List[Dict]) -> List[str]:
        """Generate related query suggestions based on collected snippets."""
        all_categories = set()
        all_patterns = set()
        all_imports = set()

        for snippet_data in snippets_data:
            metadata = snippet_data['metadata']
            all_categories.update(metadata.get('categories', []))
            all_patterns.update(metadata.get('patterns', []))
            all_imports.update(metadata.get('imports', []))

        suggestions = []

        # Category-based suggestions
        for category in list(all_categories)[:5]:
            suggestions.append(f"{category} examples")
            suggestions.append(f"{category} best practices")

        # Pattern-based suggestions
        for pattern in list(all_patterns)[:3]:
            suggestions.append(f"{pattern} implementation")

        # Import-based suggestions
        for imp in list(all_imports)[:5]:
            suggestions.append(f"{imp} tutorial")
            suggestions.append(f"{imp} advanced usage")

        return suggestions[:15]  # Limit suggestions

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Create filesystem-safe filename."""
        safe_text = re.sub(r'[^\w\s-]', '', text)
        safe_text = re.sub(r'[-\s]+', '_', safe_text)
        return safe_text[:max_length].strip('_')