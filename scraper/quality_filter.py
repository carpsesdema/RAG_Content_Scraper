# scraper/quality_filter.py

import ast
import re
from typing import List, Dict, Any


class CodeQualityFilter:
    """Filters and scores code snippets for RAG quality."""

    def __init__(self):
        self.min_lines = 3
        self.max_lines = 100
        self.min_complexity_score = 2

    def score_snippet(self, code: str, source: str = "") -> Dict[str, Any]:
        """Score a code snippet for RAG usefulness."""
        score = 0
        metadata = {
            'source': source,
            'lines': len(code.splitlines()),
            'has_docstring': False,
            'has_imports': False,
            'has_functions': False,
            'has_classes': False,
            'complexity': 0,
            'quality_issues': []
        }

        lines = code.splitlines()
        metadata['lines'] = len(lines)

        # Basic length filtering
        if metadata['lines'] < self.min_lines:
            metadata['quality_issues'].append('too_short')
            return {'score': 0, 'metadata': metadata}

        if metadata['lines'] > self.max_lines:
            metadata['quality_issues'].append('too_long')
            score -= 2

        # Check for common low-quality patterns
        code_lower = code.lower()
        if any(pattern in code_lower for pattern in ['todo', 'fixme', 'hack', 'temporary']):
            metadata['quality_issues'].append('contains_todo')
            score -= 1

        # Look for debug/print statements (might indicate example code)
        debug_patterns = ['print(', 'pprint(', 'console.log', 'debug']
        if any(pattern in code_lower for pattern in debug_patterns):
            score += 1  # Actually good for examples

        # Try AST analysis for Python code
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metadata['has_functions'] = True
                    score += 2

                    # Check for docstring
                    if (ast.get_docstring(node)):
                        metadata['has_docstring'] = True
                        score += 3

                elif isinstance(node, ast.AsyncFunctionDef):
                    metadata['has_functions'] = True
                    score += 2

                    if (ast.get_docstring(node)):
                        metadata['has_docstring'] = True
                        score += 3

                elif isinstance(node, ast.ClassDef):
                    metadata['has_classes'] = True
                    score += 3

                    if (ast.get_docstring(node)):
                        metadata['has_docstring'] = True
                        score += 2

                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    metadata['has_imports'] = True
                    score += 1

            # Calculate complexity (simplified)
            complexity = sum(1 for node in ast.walk(tree)
                             if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)))
            metadata['complexity'] = complexity
            score += min(complexity, 5)  # Cap complexity bonus

        except SyntaxError:
            # Not valid Python, but might still be useful
            metadata['quality_issues'].append('syntax_error')
            score -= 1

        # Bonus for certain sources
        if 'github' in source.lower() and metadata['has_functions']:
            score += 2
        elif 'stackoverflow' in source.lower():
            score += 1  # SO answers are often good examples

        return {'score': max(0, score), 'metadata': metadata}

    def filter_snippets(self, snippets: List[str], sources: List[str] = None) -> List[Dict]:
        """Filter and rank snippets by quality."""
        if sources is None:
            sources = [''] * len(snippets)

        scored_snippets = []
        for snippet, source in zip(snippets, sources):
            result = self.score_snippet(snippet, source)
            if result['score'] >= self.min_complexity_score:
                scored_snippets.append({
                    'code': snippet,
                    'score': result['score'],
                    'metadata': result['metadata']
                })

        # Sort by score descending
        return sorted(scored_snippets, key=lambda x: x['score'], reverse=True)