# storage/rag_exporter.py

import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class RAGExporter:
    """Export scraped code in formats optimized for RAG systems."""

    def __init__(self):
        self.export_formats = ['jsonl', 'markdown', 'xml', 'yaml']

    def export_for_rag(self, snippets_data: List[Dict],
                       output_dir: str,
                       query: str,
                       format_type: str = 'jsonl') -> str:
        """Export snippets in RAG-friendly format."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')[:50]

        if format_type == 'jsonl':
            return self._export_jsonl(snippets_data, output_path, safe_query, timestamp)
        elif format_type == 'markdown':
            return self._export_markdown(snippets_data, output_path, safe_query, timestamp)
        elif format_type == 'xml':
            return self._export_xml(snippets_data, output_path, safe_query, timestamp)
        elif format_type == 'yaml':
            return self._export_yaml(snippets_data, output_path, safe_query, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _export_jsonl(self, snippets_data: List[Dict], output_path: Path,
                      query: str, timestamp: str) -> str:
        """Export as JSONL for easy RAG ingestion."""
        filename = output_path / f"rag_snippets_{query}_{timestamp}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            for i, snippet_data in enumerate(snippets_data):
                doc = {
                    'id': f"{query}_{i}",
                    'query': query,
                    'content': snippet_data['code'],
                    'score': snippet_data.get('score', 0),
                    'metadata': {
                        'source': snippet_data['metadata'].get('source', ''),
                        'lines': snippet_data['metadata'].get('lines', 0),
                        'has_docstring': snippet_data['metadata'].get('has_docstring', False),
                        'has_functions': snippet_data['metadata'].get('has_functions', False),
                        'has_classes': snippet_data['metadata'].get('has_classes', False),
                        'complexity': snippet_data['metadata'].get('complexity', 0),
                        'timestamp': timestamp
                    },
                    'tags': self._generate_tags(snippet_data)
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        return str(filename)

    def _export_markdown(self, snippets_data: List[Dict], output_path: Path,
                         query: str, timestamp: str) -> str:
        """Export as structured markdown."""
        filename = output_path / f"rag_snippets_{query}_{timestamp}.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Code Snippets for: {query}\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Total snippets:** {len(snippets_data)}\n\n")

            for i, snippet_data in enumerate(snippets_data):
                metadata = snippet_data['metadata']
                f.write(f"## Snippet {i + 1}\n\n")
                f.write(f"**Score:** {snippet_data.get('score', 0)}\n")
                f.write(f"**Source:** {metadata.get('source', 'Unknown')}\n")
                f.write(f"**Lines:** {metadata.get('lines', 0)}\n")

                tags = self._generate_tags(snippet_data)
                if tags:
                    f.write(f"**Tags:** {', '.join(tags)}\n")

                f.write("\n```python\n")
                f.write(snippet_data['code'])
                f.write("\n```\n\n")
                f.write("---\n\n")

        return str(filename)

    def _export_xml(self, snippets_data: List[Dict], output_path: Path,
                    query: str, timestamp: str) -> str:
        """Export as XML for structured processing."""
        filename = output_path / f"rag_snippets_{query}_{timestamp}.xml"

        root = ET.Element("code_snippets")
        root.set("query", query)
        root.set("timestamp", timestamp)
        root.set("count", str(len(snippets_data)))

        for i, snippet_data in enumerate(snippets_data):
            snippet_elem = ET.SubElement(root, "snippet")
            snippet_elem.set("id", str(i))
            snippet_elem.set("score", str(snippet_data.get('score', 0)))

            # Add code content
            code_elem = ET.SubElement(snippet_elem, "code")
            code_elem.text = snippet_data['code']

            # Add metadata
            metadata_elem = ET.SubElement(snippet_elem, "metadata")
            for key, value in snippet_data['metadata'].items():
                meta_item = ET.SubElement(metadata_elem, key)
                meta_item.text = str(value)

            # Add tags
            tags_elem = ET.SubElement(snippet_elem, "tags")
            for tag in self._generate_tags(snippet_data):
                tag_elem = ET.SubElement(tags_elem, "tag")
                tag_elem.text = tag

        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        return str(filename)

    def _export_yaml(self, snippets_data: List[Dict], output_path: Path,
                     query: str, timestamp: str) -> str:
        """Export as YAML for configuration-style RAG systems."""
        filename = output_path / f"rag_snippets_{query}_{timestamp}.yaml"

        data = {
            'query': query,
            'timestamp': timestamp,
            'total_snippets': len(snippets_data),
            'snippets': []
        }

        for i, snippet_data in enumerate(snippets_data):
            snippet_entry = {
                'id': i,
                'score': snippet_data.get('score', 0),
                'code': snippet_data['code'],
                'metadata': snippet_data['metadata'],
                'tags': self._generate_tags(snippet_data)
            }
            data['snippets'].append(snippet_entry)

        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return str(filename)

    def _generate_tags(self, snippet_data: Dict) -> List[str]:
        """Generate tags for better RAG retrieval."""
        tags = []
        metadata = snippet_data['metadata']

        # Add tags based on content
        if metadata.get('has_functions'):
            tags.append('functions')
        if metadata.get('has_classes'):
            tags.append('classes')
        if metadata.get('has_docstring'):
            tags.append('documented')
        if metadata.get('has_imports'):
            tags.append('imports')

        # Add complexity tags
        complexity = metadata.get('complexity', 0)
        if complexity == 0:
            tags.append('simple')
        elif complexity <= 2:
            tags.append('moderate')
        else:
            tags.append('complex')

        # Add source tags
        source = metadata.get('source', '').lower()
        if 'github' in source:
            tags.append('github')
        elif 'stackoverflow' in source:
            tags.append('stackoverflow')
        elif 'stdlib' in source:
            tags.append('stdlib')
        elif 'additional' in source:
            tags.append('tutorial')

        return tags