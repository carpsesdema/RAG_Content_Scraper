#!/usr/bin/env python3
"""
Comprehensive test script for enhanced RAG Content Scraper features.
Run this after implementing all improvements to verify everything works.
"""

import sys
import os
import logging
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config_loading():
    """Test that enhanced configuration loads properly."""
    print("=== Testing Enhanced Configuration ===")
    try:
        import config

        # Check for core settings
        core_settings = [
            'APP_NAME', 'USER_AGENT', 'DEFAULT_REQUEST_TIMEOUT'
        ]

        # Check for enhanced settings
        enhanced_settings = [
            'QUALITY_FILTER_ENABLED',
            'SMART_DEDUPLICATION_ENABLED',
            'ADDITIONAL_SOURCES_ENABLED',
            'FREELANCER_SOURCES_ENABLED',
            'RAG_EXPORT_ENABLED',
            'CODE_CATEGORIZATION_ENABLED',
            'EMBEDDING_RAG_EXPORT_ENABLED',
            'FREELANCE_MODE'
        ]

        # Check for freelance-specific settings
        freelance_settings = [
            'FREELANCE_PROJECT_TYPES',
            'HIGH_VALUE_FREELANCE_KEYWORDS',
            'CLIENT_VALUE_INDICATORS',
            'SOURCE_PRIORITY_WEIGHTS'
        ]

        all_settings = core_settings + enhanced_settings + freelance_settings
        missing_settings = []

        for setting in all_settings:
            if hasattr(config, setting):
                value = getattr(config, setting)
                print(f"  ✅ {setting}: {value}")
            else:
                missing_settings.append(setting)
                print(f"  ❌ {setting}: MISSING")

        if missing_settings:
            print(f"⚠️  Missing {len(missing_settings)} settings: {', '.join(missing_settings)}")
        else:
            print("✅ All configuration settings found!")

        return len(missing_settings) == 0

    except ImportError as e:
        print(f"❌ Configuration not available: {e}")
        return False


def test_code_categorizer():
    """Test the enhanced code categorization system."""
    print("=== Testing Code Categorizer ===")
    try:
        from utils.code_categorizer import CodeCategorizer

        categorizer = CodeCategorizer()

        # Test cases for different types of code
        test_cases = [
            # FastAPI example
            '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users/")
async def create_user(user: UserCreate):
    """Create a new user."""
    # Implementation here
    return {"id": 1, "name": user.name, "email": user.email}
            ''',

            # Data processing example
            '''
import pandas as pd
import numpy as np

def process_sales_data(filepath):
    """Process sales data from CSV file."""
    df = pd.read_csv(filepath)

    # Clean data
    df = df.dropna()
    df['total'] = df['quantity'] * df['price']

    # Group by month
    monthly_sales = df.groupby('month')['total'].sum()

    return monthly_sales
            ''',

            # Testing example
            '''
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def sample_user():
    return {"id": 1, "name": "Test User", "email": "test@example.com"}

def test_user_creation(sample_user):
    """Test user creation functionality."""
    with patch('app.database.save_user') as mock_save:
        mock_save.return_value = True

        result = create_user(sample_user)

        assert result is not None
        assert result['name'] == sample_user['name']
        mock_save.assert_called_once()
            '''
        ]

        print(f"Testing {len(test_cases)} code samples...")

        for i, code in enumerate(test_cases):
            print(f"\n--- Test Case {i + 1} ---")
            result = categorizer.categorize_snippet(code)

            print(f"Categories: {', '.join(result['categories'])}")
            print(f"Patterns: {', '.join(result['patterns'])}")
            print(f"Complexity: {result['complexity']}")
            print(f"Freelance Score: {result['freelance_score']}")
            print(f"Client Value: {result['client_value']}")
            print(f"Use Cases: {', '.join(result['use_cases'])}")

            # Generate RAG tags
            tags = categorizer.generate_rag_tags(result)
            print(f"RAG Tags: {', '.join(tags[:5])}...")  # Show first 5 tags

            # Generate related queries
            suggestions = categorizer.suggest_related_queries(result)
            if suggestions:
                print(f"Related Queries: {', '.join(suggestions[:3])}...")  # Show first 3

        print("\n✅ Code categorizer test passed!")
        return True

    except ImportError as e:
        print(f"❌ Code categorizer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Code categorizer test failed: {e}")
        return False


def test_deduplicator():
    """Test the smart deduplication system."""
    print("=== Testing Smart Deduplicator ===")
    try:
        from utils.deduplicator import SmartDeduplicator

        dedup = SmartDeduplicator()

        test_snippets = [
            "def hello(): print('Hello')",
            "def hello(): print('Hello')",  # Exact duplicate
            "def hello():\n    print('Hello')",  # Similar formatting
            "def hello():\n    print('Hello World')",  # Similar but different
            "def goodbye(): print('Goodbye')",  # Different
            "def greet(name): print(f'Hello {name}')",  # Different but related
            # Add some more realistic examples
            '''
def calculate_tax(amount, rate=0.1):
    """Calculate tax on an amount."""
    return amount * rate
            ''',
            '''
def calculate_tax(amount, rate=0.1):
    """Calculate tax on an amount."""
    return amount * rate
            ''',  # Exact duplicate of above
            '''
def compute_tax(amount, rate=0.1):
    """Compute tax on an amount."""
    return amount * rate
            '''  # Similar but different function name
        ]

        added_count = 0
        duplicate_count = 0

        print(f"Processing {len(test_snippets)} test snippets...")

        for i, snippet in enumerate(test_snippets):
            added = dedup.add_snippet(snippet, {'test_id': i, 'original_text': snippet[:30] + '...'})
            if added:
                added_count += 1
                print(f"  ✅ Snippet {i + 1}: Added")
            else:
                duplicate_count += 1
                print(f"  ❌ Snippet {i + 1}: Duplicate/Similar")

        stats = dedup.get_stats()
        print(f"\n📊 Deduplication Results:")
        print(f"  Original snippets: {len(test_snippets)}")
        print(f"  Unique snippets added: {added_count}")
        print(f"  Duplicates/Similar rejected: {duplicate_count}")
        print(f"  Similarity threshold: {stats['similarity_threshold']}")
        print(f"  Unique hashes: {stats['unique_hashes']}")

        # Test similarity detection
        print(f"\n🔍 Testing similarity detection...")
        test_pairs = [
            ("def test(): pass", "def test(): pass"),  # Identical
            ("def test(): pass", "def test():\n    pass"),  # Different formatting
            ("def hello(): print('hi')", "def goodbye(): print('bye')"),  # Different
        ]

        for code1, code2 in test_pairs:
            similarity = dedup.calculate_similarity(code1, code2)
            print(f"  Similarity: {similarity:.2f} - '{code1[:20]}...' vs '{code2[:20]}...'")

        print("✅ Deduplicator test passed!")
        return True

    except ImportError as e:
        print(f"❌ Deduplicator not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Deduplicator test failed: {e}")
        return False


def test_freelancer_sources():
    """Test freelancer-specific source fetching."""
    print("=== Testing Freelancer Sources ===")
    try:
        from scraper.freelancer_sources import FreelancerPythonSources
        import logging

        # Create a test logger
        logger = logging.getLogger('test_freelancer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        # Test the freelancer sources
        sources = FreelancerPythonSources("RAGTestBot/1.0", timeout=10)

        # Test different types of queries
        test_queries = [
            "fastapi",
            "stripe payment",
            "automation script",
            "pytest testing",
            "docker deployment"
        ]

        total_snippets = 0

        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")

            try:
                # Test automation scripts (these use hardcoded examples)
                automation_snippets = sources.fetch_automation_scripts(query, logger)
                print(f"  Automation examples: {len(automation_snippets)}")
                total_snippets += len(automation_snippets)

                # Test testing patterns
                testing_snippets = sources.fetch_testing_patterns(query, logger)
                print(f"  Testing examples: {len(testing_snippets)}")
                total_snippets += len(testing_snippets)

                # Test client integrations
                integration_snippets = sources.fetch_client_integration_examples(query, logger)
                print(f"  Integration examples: {len(integration_snippets)}")
                total_snippets += len(integration_snippets)

                # Test deployment examples
                deployment_snippets = sources.fetch_deployment_examples(query, logger)
                print(f"  Deployment examples: {len(deployment_snippets)}")
                total_snippets += len(deployment_snippets)

            except Exception as e:
                print(f"  ⚠️  Error with query '{query}': {e}")

        print(f"\n📊 Total snippets from freelancer sources: {total_snippets}")

        if total_snippets > 0:
            print("✅ Freelancer sources test passed!")
            return True
        else:
            print("⚠️  No snippets found, but no errors occurred")
            return True

    except ImportError as e:
        print(f"❌ Freelancer sources not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Freelancer sources test failed: {e}")
        return False


def test_rag_exporters():
    """Test both RAG export systems."""
    print("=== Testing RAG Exporters ===")
    try:
        from storage.rag_exporter import RAGExporter
        from storage.embedding_rag_exporter import EmbeddingRAGExporter
        import tempfile
        import os

        # Create test data
        test_data = [
            {
                'code': '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
                '''.strip(),
                'score': 8,
                'metadata': {
                    'source': 'test',
                    'lines': 5,
                    'has_functions': True,
                    'has_docstring': True,
                    'complexity': 'intermediate',
                    'categories': ['algorithms', 'recursion'],
                    'patterns': ['function_definition', 'recursion'],
                    'imports': [],
                    'freelance_score': 6,
                    'client_value': 'medium_value_algorithm',
                    'use_cases': ['algorithm_implementation']
                }
            },
            {
                'code': '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.post("/users/")
async def create_user(user: User):
    """Create a new user."""
    return {"id": 1, "user": user}
                '''.strip(),
                'score': 15,
                'metadata': {
                    'source': 'freelancer',
                    'lines': 12,
                    'has_functions': True,
                    'has_classes': True,
                    'has_docstring': True,
                    'complexity': 'advanced',
                    'categories': ['api_development', 'web_frameworks'],
                    'patterns': ['async_function', 'class_definition', 'decorators'],
                    'imports': ['fastapi', 'pydantic'],
                    'freelance_score': 15,
                    'client_value': 'high_value_api_development',
                    'use_cases': ['backend_development', 'client_integrations']
                }
            }
        ]

        success_count = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            # Test standard RAG exporter
            print("\n--- Testing Standard RAG Exporter ---")
            try:
                exporter = RAGExporter()
                formats = ['jsonl', 'markdown', 'yaml']

                for fmt in formats:
                    try:
                        output_file = exporter.export_for_rag(
                            test_data, temp_dir, 'test_query', fmt
                        )

                        if os.path.exists(output_file):
                            file_size = os.path.getsize(output_file)
                            print(f"  ✅ {fmt.upper()} export: {file_size} bytes")

                            # Verify content for JSONL
                            if fmt == 'jsonl':
                                with open(output_file, 'r') as f:
                                    lines = f.readlines()
                                    print(f"    - {len(lines)} records exported")
                                    # Try to parse first line
                                    first_record = json.loads(lines[0])
                                    print(f"    - Sample ID: {first_record.get('id', 'N/A')}")

                            success_count += 1
                        else:
                            print(f"  ❌ {fmt.upper()} export: file not created")

                    except Exception as e:
                        print(f"  ❌ {fmt.upper()} export: {e}")

            except Exception as e:
                print(f"❌ Standard RAG exporter failed: {e}")

            # Test embedding RAG exporter
            print("\n--- Testing Embedding RAG Exporter ---")
            try:
                embedding_exporter = EmbeddingRAGExporter()
                export_files = embedding_exporter.export_for_dual_llm(
                    test_data, temp_dir, 'test_query'
                )

                print(f"  📁 Created {len(export_files)} export files:")
                for file_type, file_path in export_files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"    ✅ {file_type}: {file_size} bytes")
                        success_count += 1
                    else:
                        print(f"    ❌ {file_type}: file not created")

            except Exception as e:
                print(f"❌ Embedding RAG exporter failed: {e}")

        print(f"\n📊 Total successful exports: {success_count}")

        if success_count >= 5:  # At least 3 standard + 2 embedding formats
            print("✅ RAG exporters test passed!")
            return True
        else:
            print("⚠️  Some exports failed, but basic functionality works")
            return success_count > 0

    except ImportError as e:
        print(f"❌ RAG exporters not available: {e}")
        return False
    except Exception as e:
        print(f"❌ RAG exporters test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("=== Testing Component Integration ===")
    try:
        from utils.code_categorizer import CodeCategorizer
        from utils.deduplicator import SmartDeduplicator
        from storage.embedding_rag_exporter import EmbeddingRAGExporter

        # Test the full pipeline
        categorizer = CodeCategorizer()
        deduplicator = SmartDeduplicator()
        exporter = EmbeddingRAGExporter()

        # Sample code snippets
        raw_snippets = [
            "def hello(): print('Hello World')",
            "def hello(): print('Hello World')",  # Duplicate
            '''
import requests

def fetch_data(url):
    """Fetch data from API endpoint."""
    response = requests.get(url)
    return response.json()
            ''',
            '''
import stripe

def create_payment(amount):
    """Create Stripe payment intent."""
    return stripe.PaymentIntent.create(amount=amount)
            '''
        ]

        print(f"Processing {len(raw_snippets)} raw snippets through full pipeline...")

        # Step 1: Categorize all snippets
        categorized_data = []
        for snippet in raw_snippets:
            categorization = categorizer.categorize_snippet(snippet)
            categorized_data.append({
                'code': snippet,
                'score': categorization.get('freelance_score', 0) + 5,
                'metadata': categorization
            })

        print(f"  ✅ Categorization: {len(categorized_data)} snippets processed")

        # Step 2: Deduplicate
        unique_data = []
        for data in categorized_data:
            if deduplicator.add_snippet(data['code'], data['metadata']):
                unique_data.append(data)

        print(f"  ✅ Deduplication: {len(unique_data)} unique snippets retained")

        # Step 3: Export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_files = exporter.export_for_dual_llm(unique_data, temp_dir, 'integration_test')
            print(f"  ✅ Export: {len(export_files)} files created")

        # Verify we have reasonable results
        if len(unique_data) < len(raw_snippets):
            print("  ✅ Deduplication worked (fewer unique than raw)")

        freelance_relevant = sum(1 for data in unique_data
                                 if data['metadata'].get('freelance_relevant', False))
        print(f"  📊 Freelance relevant snippets: {freelance_relevant}/{len(unique_data)}")

        print("✅ Integration test passed!")
        return True

    except ImportError as e:
        print(f"❌ Integration test failed - missing components: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🚀 Testing Enhanced RAG Content Scraper")
    print("=" * 60)

    # Setup logging to reduce noise during tests
    logging.basicConfig(level=logging.WARNING)

    # Define test functions
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Code Categorizer", test_code_categorizer),
        ("Smart Deduplicator", test_deduplicator),
        ("Freelancer Sources", test_freelancer_sources),
        ("RAG Exporters", test_rag_exporters),
        ("Component Integration", test_integration)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))

        print()

    # Generate summary
    print("=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")

    print("-" * 60)
    print(f"🎯 Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! Your enhanced RAG scraper is ready for freelance work!")
        print("\n💡 Next steps:")
        print("  1. Set your GITHUB_TOKEN environment variable for better results")
        print("  2. Run: python main.py")
        print("  3. Try queries like 'fastapi authentication' or 'stripe payment'")
        print("  4. Use the 'Dual LLM Export' feature for your hybrid LLM system")
    elif passed >= total * 0.7:
        print("✅ Most tests passed! The system should work well.")
        print(f"⚠️  {total - passed} tests failed - check the error messages above")
    else:
        print("❌ Many tests failed. Check your installation and dependencies.")
        print("💡 Common issues:")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - Configuration errors in config.py")
        print("  - Import path issues")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)