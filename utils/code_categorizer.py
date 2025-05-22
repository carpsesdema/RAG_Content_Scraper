# utils/code_categorizer.py

import ast
import re
from typing import Dict, List, Set
from collections import defaultdict


class CodeCategorizer:
    """Categorize code snippets for better RAG retrieval."""

    def __init__(self):
        self.categories = {
            # Architectural patterns
            'design_patterns': ['singleton', 'factory', 'observer', 'decorator', 'strategy'],
            'async_programming': ['async', 'await', 'asyncio', 'coroutine', 'future'],
            'api_development': ['fastapi', 'flask', 'django', 'rest', 'endpoint', 'route'],
            'data_processing': ['pandas', 'numpy', 'scipy', 'dataframe', 'array'],
            'database': ['sqlalchemy', 'pymongo', 'sqlite', 'postgresql', 'cursor'],
            'testing': ['pytest', 'unittest', 'mock', 'fixture', 'test_'],
            'web_scraping': ['requests', 'beautifulsoup', 'selenium', 'scrapy'],
            'machine_learning': ['sklearn', 'tensorflow', 'pytorch', 'keras', 'model'],
            'data_validation': ['pydantic', 'marshmallow', 'cerberus', 'schema'],
            'cli_tools': ['click', 'argparse', 'typer', 'command', 'parser'],
            'configuration': ['configparser', 'yaml', 'json', 'settings', 'config'],
            'logging': ['logging', 'logger', 'log', 'debug', 'info'],
            'file_operations': ['pathlib', 'os.path', 'shutil', 'glob', 'file'],
            'error_handling': ['try', 'except', 'finally', 'raise', 'exception'],
            'performance': ['multiprocessing', 'threading', 'concurrent', 'profile'],
            'security': ['hashlib', 'secrets', 'cryptography', 'jwt', 'auth'],
            'utilities': ['datetime', 'collections', 'itertools', 'functools', 'operator'],
            'deployment': ['docker', 'kubernetes', 'gunicorn', 'uvicorn', 'systemd'],
            'monitoring': ['prometheus', 'grafana', 'sentry', 'logging', 'metrics'],
            'caching': ['redis', 'memcached', 'cache', 'lru_cache', 'cached'],
            'message_queues': ['celery', 'rabbitmq', 'kafka', 'queue', 'task'],
            'client_integrations': ['stripe', 'twilio', 'sendgrid', 'slack', 'discord']
        }

        self.complexity_patterns = {
            'beginner': ['print', 'input', 'len', 'range', 'for', 'if'],
            'intermediate': ['class', 'def', 'import', 'try', 'with'],
            'advanced': ['__init__', '__enter__', '__exit__', 'metaclass', 'descriptor'],
            'expert': ['__new__', '__getattribute__', 'exec', 'eval', 'compile']
        }

        self.freelance_indicators = {
            'high_value': ['fastapi', 'django', 'stripe', 'aws', 'docker', 'pytest'],
            'automation': ['selenium', 'schedule', 'cron', 'automation', 'script'],
            'data_work': ['pandas', 'numpy', 'csv', 'excel', 'etl', 'pipeline'],
            'integrations': ['api', 'webhook', 'oauth', 'jwt', 'rest', 'graphql']
        }

    def categorize_snippet(self, code: str, metadata: Dict = None) -> Dict:
        """Categorize a code snippet with detailed analysis."""
        if metadata is None:
            metadata = {}

        categories = set()
        imports_used = set()
        patterns_found = set()
        complexity_level = 'beginner'
        freelance_score = 0

        # Extract imports and functions
        try:
            tree = ast.parse(code)

            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_used.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_used.add(node.module.split('.')[0])

                # Analyze patterns
                if isinstance(node, ast.ClassDef):
                    patterns_found.add('class_definition')
                    if any(base.id in ['ABC', 'BaseModel', 'Enum'] for base in node.bases if hasattr(base, 'id')):
                        patterns_found.add('inheritance')
                    if any(decorator.id == 'dataclass' for decorator in node.decorator_list if
                           hasattr(decorator, 'id')):
                        patterns_found.add('dataclass')
                elif isinstance(node, ast.FunctionDef):
                    patterns_found.add('function_definition')
                    if node.name.startswith('_'):
                        patterns_found.add('private_method')
                    if node.decorator_list:
                        patterns_found.add('decorators')
                    if any(decorator.id in ['property', 'staticmethod', 'classmethod'] for decorator in
                           node.decorator_list if hasattr(decorator, 'id')):
                        patterns_found.add('special_methods')
                elif isinstance(node, ast.AsyncFunctionDef):
                    patterns_found.add('async_function')
                elif isinstance(node, ast.With):
                    patterns_found.add('context_manager')
                elif isinstance(node, ast.Try):
                    patterns_found.add('exception_handling')
                elif isinstance(node, ast.ListComp) or isinstance(node, ast.DictComp):
                    patterns_found.add('comprehension')

        except SyntaxError:
            # Still analyze as text for non-Python or incomplete code
            pass

        # Text-based pattern matching
        code_lower = code.lower()

        # Categorize based on imports and content
        for category, keywords in self.categories.items():
            if any(keyword in code_lower or keyword in imports_used for keyword in keywords):
                categories.add(category)

        # Determine complexity
        for level, keywords in self.complexity_patterns.items():
            if any(keyword in code_lower for keyword in keywords):
                complexity_level = level

        # Special pattern detection
        if 'class' in code_lower and 'def __init__' in code_lower:
            patterns_found.add('class_with_constructor')
        if re.search(r'@\w+', code):
            patterns_found.add('decorators')
        if 'yield' in code_lower:
            patterns_found.add('generator')
        if re.search(r'f["\'].*{.*}.*["\']', code):
            patterns_found.add('f_strings')
        if 'lambda' in code_lower:
            patterns_found.add('lambda_functions')
        if re.search(r'async\s+def', code):
            patterns_found.add('async_function')
        if re.search(r'with\s+\w+', code):
            patterns_found.add('context_manager')

        # Calculate freelance value score
        for value_type, keywords in self.freelance_indicators.items():
            if any(keyword in code_lower or keyword in imports_used for keyword in keywords):
                if value_type == 'high_value':
                    freelance_score += 3
                else:
                    freelance_score += 2

        # Use case detection for freelance work
        use_cases = set()
        if any(cat in categories for cat in ['api_development', 'database']):
            use_cases.add('backend_development')
        if any(cat in categories for cat in ['web_scraping', 'data_processing']):
            use_cases.add('data_engineering')
        if 'testing' in categories:
            use_cases.add('test_automation')
        if any(cat in categories for cat in ['cli_tools', 'file_operations']):
            use_cases.add('automation_scripts')
        if 'machine_learning' in categories:
            use_cases.add('ml_development')
        if any(cat in categories for cat in ['client_integrations', 'api_development']):
            use_cases.add('client_integrations')
        if any(cat in categories for cat in ['deployment', 'monitoring']):
            use_cases.add('devops_automation')

        return {
            'categories': list(categories),
            'imports': list(imports_used),
            'patterns': list(patterns_found),
            'complexity': complexity_level,
            'use_cases': list(use_cases),
            'freelance_relevant': len(use_cases) > 0,
            'freelance_score': freelance_score,
            'snippet_type': self._determine_snippet_type(patterns_found, categories),
            'client_value': self._assess_client_value(categories, patterns_found, imports_used)
        }

    def _determine_snippet_type(self, patterns: Set[str], categories: Set[str]) -> str:
        """Determine the primary type of code snippet."""
        if 'class_definition' in patterns:
            return 'class_implementation'
        elif 'function_definition' in patterns:
            return 'function_implementation'
        elif 'api_development' in categories:
            return 'api_endpoint'
        elif 'testing' in categories:
            return 'test_case'
        elif 'data_processing' in categories:
            return 'data_script'
        elif 'configuration' in categories:
            return 'configuration_setup'
        elif 'client_integrations' in categories:
            return 'integration_example'
        else:
            return 'code_snippet'

    def _assess_client_value(self, categories: Set[str], patterns: Set[str], imports: Set[str]) -> str:
        """Assess the potential client value of this code."""
        high_value_indicators = {
            'payment_processing': ['stripe', 'paypal', 'payment'],
            'communication': ['twilio', 'sendgrid', 'slack', 'discord'],
            'cloud_services': ['aws', 'azure', 'gcp', 'boto3'],
            'data_analytics': ['pandas', 'numpy', 'matplotlib', 'plotly'],
            'web_automation': ['selenium', 'scrapy', 'requests'],
            'api_development': ['fastapi', 'django', 'flask'],
            'testing': ['pytest', 'unittest', 'selenium']
        }

        for value_type, indicators in high_value_indicators.items():
            if any(indicator in categories or indicator in imports for indicator in indicators):
                return f'high_value_{value_type}'

        if any(cat in categories for cat in ['automation_scripts', 'data_processing']):
            return 'medium_value_automation'
        elif any(cat in categories for cat in ['utilities', 'file_operations']):
            return 'utility_code'
        else:
            return 'general_purpose'

    def generate_rag_tags(self, categorization: Dict) -> List[str]:
        """Generate optimized tags for RAG retrieval."""
        tags = []

        # Add category tags
        for category in categorization['categories']:
            tags.append(f"category:{category}")

        # Add complexity tag
        tags.append(f"complexity:{categorization['complexity']}")

        # Add pattern tags
        for pattern in categorization['patterns']:
            tags.append(f"pattern:{pattern}")

        # Add use case tags
        for use_case in categorization['use_cases']:
            tags.append(f"usecase:{use_case}")

        # Add import tags for dependency tracking
        for imp in categorization['imports'][:5]:  # Limit to avoid tag explosion
            tags.append(f"uses:{imp}")

        # Add snippet type
        tags.append(f"type:{categorization['snippet_type']}")

        # Add freelance relevance
        if categorization['freelance_relevant']:
            tags.append("freelance:relevant")
            tags.append(f"freelance_score:{categorization['freelance_score']}")

        # Add client value
        tags.append(f"client_value:{categorization['client_value']}")

        return tags

    def suggest_related_queries(self, categorization: Dict) -> List[str]:
        """Suggest related search queries based on categorization."""
        suggestions = []

        categories = categorization['categories']
        patterns = categorization['patterns']
        imports = categorization['imports']

        # Category-based suggestions
        if 'api_development' in categories:
            suggestions.extend(['fastapi middleware', 'rest api best practices', 'api authentication'])
        if 'data_processing' in categories:
            suggestions.extend(['pandas optimization', 'data pipeline patterns', 'etl automation'])
        if 'testing' in categories:
            suggestions.extend(['pytest fixtures', 'test automation', 'mock testing'])
        if 'web_scraping' in categories:
            suggestions.extend(['selenium automation', 'requests session management', 'scraping best practices'])

        # Import-based suggestions
        for imp in imports[:3]:
            suggestions.append(f"{imp} examples")
            suggestions.append(f"{imp} best practices")

        # Pattern-based suggestions
        if 'async_function' in patterns:
            suggestions.extend(['asyncio patterns', 'async best practices'])
        if 'decorators' in patterns:
            suggestions.extend(['python decorators', 'decorator patterns'])

        return list(set(suggestions))  # Remove duplicates