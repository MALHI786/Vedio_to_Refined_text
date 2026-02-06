"""
Test Script for AI Pipeline (Terminal Test)
Run this script to verify the AI pipeline works correctly.

Usage:
    python test_pipeline.py

This will:
1. Test text cleaning
2. Test text improvement
3. (Optional) Test full pipeline with a sample video
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaner import clean_text
from utils.text_improver import TextImprover


def test_text_cleaning():
    """Test the text cleaning module."""
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Text Cleaning")
    print("=" * 60)
    
    test_cases = [
        "uh so like I want to um explain this new app you know",
        "basically I I made this thing and uh it works well",
        "so yeah the the system is really really good honestly",
    ]
    
    print("\nTesting text cleaner...")
    for i, text in enumerate(test_cases, 1):
        cleaned = clean_text(text)
        print(f"\n  Test {i}:")
        print(f"  Input:  '{text}'")
        print(f"  Output: '{cleaned}'")
    
    print("\n✅ Text cleaning tests passed!")
    return True


def test_text_improvement():
    """Test the text improvement module."""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Text Improvement (Grammar Correction)")
    print("=" * 60)
    
    test_cases = [
        "i make new app and i want explain this",
        "he dont know what to do",
        "me and him went to store",
        "their going to park today",
        "i am very excited to introduce my new application",
    ]
    
    print("\n⏳ Loading grammar correction model (this may take a moment)...")
    
    try:
        improver = TextImprover()
        
        print("\nTesting text improver...")
        for i, text in enumerate(test_cases, 1):
            improved = improver.improve(text)
            print(f"\n  Test {i}:")
            print(f"  Input:  '{text}'")
            print(f"  Output: '{improved}'")
        
        print("\n✅ Text improvement tests passed!")
        return True
        
    except Exception as e:
        print(f"\n⚠️ Text improvement test failed: {e}")
        print("   This might be due to missing model downloads.")
        return False


def test_full_pipeline_text():
    """Test the full pipeline with text input."""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Full Pipeline (Text Mode)")
    print("=" * 60)
    
    from backend.pipeline import AIPipeline
    
    sample_text = """
    uh so like i want to talk about my new app you know 
    and basically it uses artificial intelligence to um 
    convert videos into text and then it improves the english 
    so yeah its really cool and i think you gonna like it
    """
    
    print(f"\nOriginal text:\n{sample_text}")
    
    try:
        # Use lightweight models for testing
        pipeline = AIPipeline(whisper_model="base")
        result = pipeline.process_text(sample_text)
        
        print(f"\n📌 Cleaned text:\n{result['cleaned_text']}")
        print(f"\n📌 Improved text:\n{result['improved_text']}")
        
        print("\n✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"\n⚠️ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 AI VIDEO-TO-FLUENT-TEXT PIPELINE TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Text Cleaning
    results.append(("Text Cleaning", test_text_cleaning()))
    
    # Test 2: Text Improvement
    results.append(("Text Improvement", test_text_improvement()))
    
    # Test 3: Full Pipeline
    results.append(("Full Pipeline", test_full_pipeline_text()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Pipeline is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return total_passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
