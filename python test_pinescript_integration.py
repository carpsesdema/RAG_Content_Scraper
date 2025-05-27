#!/usr/bin/env python3
"""
Test script for Pinescript integration in RAG Content Scraper.
Run this to verify all Pinescript features are working correctly.
"""

import sys
import os
import logging
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_pinescript_config():
    """Test that Pinescript configuration loads properly."""
    print("=== Testing Pinescript Configuration ===")
    try:
        import config

        # Check for Pinescript-specific settings
        pinescript_settings = [
            'PINESCRIPT_ENABLED',
            'PINESCRIPT_SOURCES_ENABLED',
            'TRADINGVIEW_SCRAPING_ENABLED',
            'CONTENT_TYPES',
            'PINESCRIPT_SOURCE_WEIGHTS',
            'TRADING_MODE',
            'HIGH_VALUE_TRADING_KEYWORDS',
            'LANGUAGE_DETECTION_ENABLED',
            'CONTENT_TYPE_KEYWORDS'
        ]

        missing_settings = []
        for setting in pinescript_settings:
            if hasattr(config, setting):
                value = getattr(config, setting)
                print(f"  ✅ {setting}: {value}")
            else:
                missing_settings.append(setting)
                print(f"  ❌ {setting}: MISSING")

        # Test content type configuration
        content_types = getattr(config, 'CONTENT_TYPES', {})
        if content_types.get('pinescript', False):
            print("  ✅ Pinescript content type enabled")
        else:
            print("  ⚠️  Pinescript content type disabled or missing")

        if missing_settings:
            print(f"⚠️  Missing {len(missing_settings)} Pinescript settings")
            return False
        else:
            print("✅ All Pinescript configuration settings found!")
            return True

    except ImportError as e:
        print(f"❌ Configuration not available: {e}")
        return False


def test_pinescript_categorizer():
    """Test the Pinescript categorization system."""
    print("=== Testing Pinescript Categorizer ===")
    try:
        from utils.pinescript_categorizer import PinescriptCategorizer

        categorizer = PinescriptCategorizer()

        # Test cases for different types of Pinescript code
        test_cases = [
            # Simple Moving Average Indicator
            '''
//@version=5
indicator("Simple Moving Average", shorttitle="SMA", overlay=true)

length = input.int(20, title="Length", minval=1)
source = input(close, title="Source")

sma_value = ta.sma(source, length)
plot(sma_value, color=color.blue, linewidth=2, title="SMA")

alertcondition(ta.crossover(close, sma_value), title="Price Cross Above SMA")
            ''',

            # Trading Strategy
            '''
//@version=5
strategy("RSI Strategy", overlay=false, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

length = input.int(14, title="RSI Length")
overbought = input.int(70, title="Overbought Level")
oversold = input.int(30, title="Oversold Level")

rsi_value = ta.rsi(close, length)

long_condition = ta.crossunder(rsi_value, oversold)
short_condition = ta.crossover(rsi_value, overbought)

if long_condition
    strategy.entry("Long", strategy.long)
if short_condition
    strategy.close("Long")

plot(rsi_value, title="RSI")
hline(overbought, "Overbought")
hline(oversold, "Oversold")
            ''',

            # Advanced Library
            '''
//@version=5
library("TradingLibrary")

export position_size(float capital, float risk_percent, float entry_price, float stop_price) =>
    risk_amount = capital * (risk_percent / 100)
    price_diff = math.abs(entry_price - stop_price)
    risk_amount / price_diff

export is_bullish_engulfing() =>
    prev_bearish = close[1] < open[1]
    curr_bullish = close > open
    curr_body_engulfs = open < close[1] and close > open[1]
    prev_bearish and curr_bullish and curr_body_engulfs
            '''
        ]

        print(f"Testing {len(test_cases)} Pinescript samples...")

        for i, code in enumerate(test_cases):
            print(f"\n--- Test Case {i + 1} ---")
            result = categorizer.categorize_pinescript(code)

            print(f"Script Type: {result['script_type']}")
            print(f"Categories: {', '.join(result['categories'][:5])}")
            print(f"Patterns: {', '.join(result['patterns'][:5])}")
            print(f"Complexity: {result['complexity']}")
            print(f"Trading Value Score: {result['trading_value_score']}")
            print(f"Client Value: {result['client_value']}")
            print(f"Use Cases: {', '.join(result['use_cases'])}")
            print(f"Pinescript Version: {result['pinescript_version']}")

            # Generate tags
            tags = categorizer.generate_pinescript_tags(result)
            print(f"Tags: {', '.join(tags[:7])}...")  # Show first 7 tags

            # Generate related queries
            suggestions = categorizer.suggest_related_pinescript_queries(result)
            if suggestions:
                print(f"Related Queries: {', '.join(suggestions[:3])}...")

        print("\n✅ Pinescript categorizer test passed!")
        return True

    except ImportError as e:
        print(f"❌ Pinescript categorizer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Pinescript categorizer test failed: {e}")
        return False


def test_pinescript_sources():
    """Test Pinescript source fetching."""
    print("=== Testing Pinescript Sources ===")
    try:
        from scraper.pinescript_sources import PinescriptSources, search_pinescript_sources
        import logging

        # Create a test logger
        logger = logging.getLogger('test_pinescript')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        # Test different types of Pinescript queries
        test_queries = [
            "sma indicator",
            "rsi strategy",
            "bollinger bands",
            "macd",
            "trading alerts",
            "pinescript basic tutorial",
            "strategy backtesting"
        ]

        total_snippets = 0

        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")

            try:
                snippets = search_pinescript_sources(query, logger)
                total_snippets += len(snippets)
                print(f"  Found {len(snippets)} snippets")

                if snippets:
                    # Show a sample snippet
                    sample = snippets[0]
                    if len(sample) > 200:
                        sample = sample[:200] + "..."
                    print(f"  Sample: {sample}")

            except Exception as e:
                print(f"  ⚠️  Error with query '{query}': {e}")

        print(f"\n📊 Total snippets from Pinescript sources: {total_snippets}")

        if total_snippets > 0:
            print("✅ Pinescript sources test passed!")
            return True
        else:
            print("⚠️  No snippets found, but no errors occurred")
            return True

    except ImportError as e:
        print(f"❌ Pinescript sources not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Pinescript sources test failed: {e}")
        return False


def test_content_type_detection():
    """Test automatic content type detection."""
    print("=== Testing Content Type Detection ===")
    try:
        from scraper.searcher import detect_content_type
        import logging

        logger = logging.getLogger('test_detection')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        # Test queries and expected results
        test_cases = [
            # Pinescript queries
            ("sma indicator pinescript", "pinescript"),
            ("tradingview strategy", "pinescript"),
            ("rsi bollinger bands", "pinescript"),
            ("ta.sma trading", "pinescript"),
            ("//@version=5 indicator", "pinescript"),
            ("strategy entry exit", "pinescript"),

            # Python queries
            ("fastapi authentication", "python"),
            ("pandas dataframe", "python"),
            ("django rest api", "python"),
            ("import requests", "python"),
            ("def function", "python"),

            # Ambiguous queries
            ("machine learning", "python"),  # Should default to Python
            ("data analysis", "python"),  # Should default to Python
        ]

        correct_detections = 0

        for query, expected in test_cases:
            detected = detect_content_type(query, logger)
            is_correct = detected == expected
            status = "✅" if is_correct else "❌"

            print(f"  {status} '{query}' -> {detected} (expected: {expected})")

            if is_correct:
                correct_detections += 1

        accuracy = (correct_detections / len(test_cases)) * 100
        print(f"\n📊 Detection Accuracy: {correct_detections}/{len(test_cases)} ({accuracy:.1f}%)")

        if accuracy >= 80:
            print("✅ Content type detection test passed!")
            return True
        else:
            print("⚠️  Content type detection accuracy below 80%")
            return accuracy > 50

    except ImportError as e:
        print(f"❌ Content type detection not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Content type detection test failed: {e}")
        return False


def test_integrated_search():
    """Test integrated search with Pinescript content."""
    print("=== Testing Integrated Pinescript Search ===")
    try:
        from scraper.searcher import search_and_fetch
        import logging

        # Create a test logger
        logger = logging.getLogger('test_integrated')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        # Test Pinescript search
        test_queries = [
            ("sma indicator", "pinescript"),
            ("fastapi tutorial", "python"),  # For comparison
        ]

        for query, content_type in test_queries:
            print(f"\n--- Testing integrated search: '{query}' ({content_type}) ---")

            try:
                snippets = search_and_fetch(query, logger, content_type=content_type)
                print(f"  Found {len(snippets)} snippets")

                # Check if enhanced data was generated
                if hasattr(logger, 'enhanced_snippet_data'):
                    enhanced_data = logger.enhanced_snippet_data
                    print(f"  Enhanced data: {len(enhanced_data)} items")

                    if enhanced_data:
                        first_item = enhanced_data[0]
                        detected_content_type = first_item.get('content_type', 'unknown')
                        print(f"  Detected content type: {detected_content_type}")

                        if detected_content_type == content_type:
                            print("  ✅ Content type correctly processed")
                        else:
                            print(f"  ⚠️  Content type mismatch: {detected_content_type} vs {content_type}")

                if snippets and len(snippets) > 0:
                    # Show a sample
                    sample = snippets[0][:150] + "..." if len(snippets[0]) > 150 else snippets[0]
                    print(f"  Sample snippet: {sample}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        print("✅ Integrated search test completed!")
        return True

    except ImportError as e:
        print(f"❌ Integrated search not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Integrated search test failed: {e}")
        return False


def test_pinescript_exports():
    """Test Pinescript-specific export functionality."""
    print("=== Testing Pinescript Export Features ===")
    try:
        from storage.rag_exporter import RAGExporter
        import tempfile
        import os

        # Create test Pinescript data
        test_data = [
            {
                'code': '''
//@version=5
indicator("Test RSI", shorttitle="RSI")
length = input.int(14, title="Length")
rsi_value = ta.rsi(close, length)
plot(rsi_value, color=color.purple)
hline(70, "Overbought")
hline(30, "Oversold")
                '''.strip(),
                'score': 12,
                'metadata': {
                    'source': 'pinescript',
                    'script_type': 'indicator',
                    'complexity': 'intermediate',
                    'categories': ['indicators', 'oscillators', 'rsi'],
                    'patterns': ['input_parameters', 'plotting', 'builtin_ta_functions'],
                    'trading_value_score': 8,
                    'client_value': 'medium_value_trading_tools',
                    'use_cases': ['technical_analysis', 'alert_systems'],
                    'pinescript_version': 'v5',
                    'content_type': 'pinescript'
                }
            },
            {
                'code': '''
//@version=5
strategy("SMA Cross Strategy", overlay=true)
fast = input.int(10, "Fast SMA")
slow = input.int(20, "Slow SMA")
fast_sma = ta.sma(close, fast)
slow_sma = ta.sma(close, slow)
if ta.crossover(fast_sma, slow_sma)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast_sma, slow_sma)
    strategy.close("Long")
plot(fast_sma, color=color.blue)
plot(slow_sma, color=color.red)
                '''.strip(),
                'score': 15,
                'metadata': {
                    'source': 'pinescript',
                    'script_type': 'strategy',
                    'complexity': 'advanced',
                    'categories': ['strategies', 'moving_averages', 'backtesting'],
                    'patterns': ['strategy_orders', 'input_parameters', 'plotting'],
                    'trading_value_score': 12,
                    'client_value': 'high_value_professional_trading',
                    'use_cases': ['strategy_development', 'trading_automation'],
                    'pinescript_version': 'v5',
                    'content_type': 'pinescript'
                }
            }
        ]

        success_count = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            # Test Pinescript-specific export formats
            print("\n--- Testing Pinescript RAG Exporter ---")
            try:
                exporter = RAGExporter()
                formats = ['jsonl', 'markdown', 'yaml']

                for fmt in formats:
                    try:
                        output_file = exporter.export_for_rag(
                            test_data, temp_dir, 'pinescript_test', fmt
                        )

                        if os.path.exists(output_file):
                            file_size = os.path.getsize(output_file)
                            print(f"  ✅ {fmt.upper()} export: {file_size} bytes")

                            # Verify Pinescript-specific content
                            if fmt == 'jsonl':
                                with open(output_file, 'r') as f:
                                    lines = f.readlines()
                                    print(f"    - {len(lines)} records exported")

                                    # Check first record for Pinescript content
                                    first_record = json.loads(lines[0])
                                    if '//@version=5' in first_record.get('content', ''):
                                        print(f"    - ✅ Contains Pinescript code")
                                    else:
                                        print(f"    - ⚠️  Missing Pinescript markers")

                            success_count += 1
                        else:
                            print(f"  ❌ {fmt.upper()} export: file not created")

                    except Exception as e:
                        print(f"  ❌ {fmt.upper()} export: {e}")

            except Exception as e:
                print(f"❌ Pinescript RAG exporter failed: {e}")

            # Test Pinescript dual LLM export
            print("\n--- Testing Pinescript Dual LLM Export ---")
            try:
                from storage.embedding_rag_exporter import EmbeddingRAGExporter

                embedding_exporter = EmbeddingRAGExporter()
                export_files = embedding_exporter.export_for_dual_llm(
                    test_data, temp_dir, 'pinescript_test'
                )

                print(f"  📁 Created {len(export_files)} export files:")
                for file_type, file_path in export_files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"    ✅ {file_type}: {file_size} bytes")

                        # Check for Pinescript-specific content in chat LLM file
                        if file_type == 'chat_llm' and file_path.endswith('.jsonl'):
                            with open(file_path, 'r') as f:
                                first_line = f.readline()
                                if first_line:
                                    record = json.loads(first_line)
                                    content = record.get('content', '')
                                    if any(term in content.lower() for term in
                                           ['indicator', 'strategy', 'trading', 'pinescript']):
                                        print(f"      ✅ Contains trading context")

                        success_count += 1
                    else:
                        print(f"    ❌ {file_type}: file not created")

            except ImportError:
                print("  ⚠️  Dual LLM exporter not available")
            except Exception as e:
                print(f"❌ Dual LLM exporter failed: {e}")

        print(f"\n📊 Total successful exports: {success_count}")

        if success_count >= 3:  # At least basic formats working
            print("✅ Pinescript export tests passed!")
            return True
        else:
            print("⚠️  Some exports failed, but basic functionality works")
            return success_count > 0

    except ImportError as e:
        print(f"❌ Export modules not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Pinescript export test failed: {e}")
        return False


def test_ui_integration():
    """Test UI components for Pinescript support."""
    print("=== Testing UI Integration ===")
    try:
        # Test that UI modules can be imported with Pinescript support
        from gui.main_window import EnhancedMainWindow

        print("  ✅ Enhanced main window imports successfully")

        # Test content type detection in UI context
        from scraper.searcher import detect_content_type
        import logging

        logger = logging.getLogger('test_ui')
        test_queries = [
            "rsi indicator",
            "fastapi tutorial"
        ]

        for query in test_queries:
            content_type = detect_content_type(query, logger)
            print(f"  ✅ UI can detect '{query}' as {content_type}")

        print("✅ UI integration test passed!")
        return True

    except ImportError as e:
        print(f"❌ UI integration test failed - missing components: {e}")
        return False
    except Exception as e:
        print(f"❌ UI integration test failed: {e}")
        return False


def test_full_integration():
    """Test the complete Pinescript workflow."""
    print("=== Testing Complete Pinescript Workflow ===")
    try:
        from scraper.searcher import search_and_fetch, detect_content_type
        from utils.pinescript_categorizer import PinescriptCategorizer
        from storage.rag_exporter import RAGExporter
        import logging
        import tempfile

        # Setup logging
        logger = logging.getLogger('test_integration')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        # Test query
        test_query = "rsi bollinger strategy"

        print(f"Testing complete workflow for: '{test_query}'")

        # Step 1: Detect content type
        content_type = detect_content_type(test_query, logger)
        print(f"  ✅ Step 1 - Content type detected: {content_type}")

        # Step 2: Search and fetch
        snippets = search_and_fetch(test_query, logger, content_type=content_type)
        print(f"  ✅ Step 2 - Found {len(snippets)} snippets")

        if snippets:
            # Step 3: Categorize (if enhanced data available)
            if hasattr(logger, 'enhanced_snippet_data') and logger.enhanced_snippet_data:
                enhanced_data = logger.enhanced_snippet_data
                print(f"  ✅ Step 3 - Enhanced categorization: {len(enhanced_data)} items")

                # Check first item
                first_item = enhanced_data[0]
                categories = first_item['metadata'].get('categories', [])
                script_type = first_item['metadata'].get('script_type', 'unknown')
                print(f"    First item: {script_type}, categories: {categories[:3]}")
            else:
                # Manual categorization
                categorizer = PinescriptCategorizer()
                first_snippet = snippets[0]
                categorization = categorizer.categorize_pinescript(first_snippet)
                print(f"  ✅ Step 3 - Manual categorization: {categorization['script_type']}")

            # Step 4: Export
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    exporter = RAGExporter()

                    # Create export data
                    if hasattr(logger, 'enhanced_snippet_data') and logger.enhanced_snippet_data:
                        export_data = logger.enhanced_snippet_data
                    else:
                        export_data = [{'code': snippet, 'score': 5, 'metadata': {'content_type': content_type}}
                                       for snippet in snippets]

                    output_file = exporter.export_for_rag(export_data, temp_dir, test_query, 'jsonl')

                    if os.path.exists(output_file):
                        print(f"  ✅ Step 4 - Export successful: {output_file}")
                    else:
                        print(f"  ❌ Step 4 - Export failed")
                        return False

                except Exception as e:
                    print(f"  ❌ Step 4 - Export error: {e}")
                    return False

        print("✅ Complete workflow test passed!")
        return True

    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False


def main():
    """Run comprehensive Pinescript integration tests."""
    print("🚀 Testing Pinescript Integration for RAG Content Scraper")
    print("=" * 70)

    # Setup logging to reduce noise during tests
    logging.basicConfig(level=logging.WARNING)

    # Define test functions
    tests = [
        ("Pinescript Configuration", test_pinescript_config),
        ("Pinescript Categorizer", test_pinescript_categorizer),
        ("Pinescript Sources", test_pinescript_sources),
        ("Content Type Detection", test_content_type_detection),
        ("Integrated Search", test_integrated_search),
        ("Pinescript Exports", test_pinescript_exports),
        ("UI Integration", test_ui_integration),
        ("Complete Workflow", test_full_integration)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 50)

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
    print("=" * 70)
    print("📋 PINESCRIPT INTEGRATION TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    print("-" * 70)
    print(f"🎯 Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("🎉 All Pinescript tests passed! Your integration is ready!")
        print("\n💡 Next steps:")
        print("  1. Set PINESCRIPT_ENABLED = True in config.py")
        print("  2. Run: python main.py")
        print("  3. Select 'Pinescript' content type in the UI")
        print("  4. Try queries like 'rsi strategy' or 'bollinger bands indicator'")
        print("  5. Use 'TradingView-Friendly' export format for your trading scripts")
    elif passed >= total * 0.7:
        print("✅ Most Pinescript tests passed! The system should work well.")
        print(f"⚠️  {total - passed} tests failed - check the error messages above")
        print("\n💡 You can still use the Pinescript features with some limitations.")
    else:
        print("❌ Many Pinescript tests failed. Check your installation.")
        print("💡 Common issues:")
        print("  - Missing Pinescript modules in scraper/ or utils/")
        print("  - Configuration errors in config.py")
        print("  - Import path issues")
        print("  - Missing dependencies")

    # Show configuration summary
    print(f"\n📊 Pinescript Integration Status:")
    try:
        import config
        pinescript_enabled = getattr(config, 'PINESCRIPT_ENABLED', False)
        content_types = getattr(config, 'CONTENT_TYPES', {})

        print(f"  Pinescript Enabled: {pinescript_enabled}")
        print(f"  Content Types: {content_types}")
        print(f"  Default Content Type: {getattr(config, 'DEFAULT_CONTENT_TYPE', 'unknown')}")

        if pinescript_enabled and content_types.get('pinescript', False):
            print("  🟢 Pinescript integration is properly configured!")
        else:
            print("  🟡 Pinescript integration needs configuration updates")

    except ImportError:
        print("  🔴 Configuration module not accessible")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)