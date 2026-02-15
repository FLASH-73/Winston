"""Quick Claude API diagnostic — run: python test_claude.py

Tests your API key, model, and response format independently of WINSTON.
Verifies that text responses come back reliably.
"""

from config import ANTHROPIC_API_KEY, FAST_MODEL, SMART_MODEL, get_conversation_prompt

print(f"API key: {ANTHROPIC_API_KEY[:12]}...{ANTHROPIC_API_KEY[-4:]}" if ANTHROPIC_API_KEY else "API key: NOT SET!")
print(f"Fast model: {FAST_MODEL}")
print(f"Smart model: {SMART_MODEL}")
print()

try:
    import anthropic

    print(f"anthropic SDK version: {anthropic.__version__}")
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    exit(1)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def dump_response(response, label=""):
    """Print detailed diagnostics for a Claude response."""
    if label:
        print(f"  [{label}]")
    if response.content:
        for i, block in enumerate(response.content):
            print(
                f"  content[{i}]: type={block.type}, text={repr(block.text[:80]) if hasattr(block, 'text') and block.text else 'N/A'}"
            )
    else:
        print("  NO CONTENT BLOCKS!")
    print(f"  stop_reason: {response.stop_reason}")
    print(f"  usage: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
    print()


# Test 1: Simple text request (Fast model)
print("=" * 60)
print(f"Test 1: Simple text request ({FAST_MODEL})")
print("=" * 60)
try:
    response = client.messages.create(
        model=FAST_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    dump_response(response, "Fast model")
    test1_ok = response.content and response.content[0].text
except Exception as e:
    print(f"  FAILED: {e}")
    test1_ok = False

# Test 2: Text with system prompt (Fast model)
print("=" * 60)
print(f"Test 2: With system prompt ({FAST_MODEL})")
print("=" * 60)
try:
    response2 = client.messages.create(
        model=FAST_MODEL,
        max_tokens=100,
        system=get_conversation_prompt(),
        messages=[{"role": "user", "content": "Hey Winston, what can you do?"}],
    )
    dump_response(response2, "Fast + system prompt")
    test2_ok = response2.content and response2.content[0].text
except Exception as e:
    print(f"  FAILED: {e}")
    test2_ok = False

# Test 3: Smart model test
print("=" * 60)
print(f"Test 3: Smart model ({SMART_MODEL})")
print("=" * 60)
try:
    response3 = client.messages.create(
        model=SMART_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
    )
    dump_response(response3, "Smart model")
    test3_ok = response3.content and response3.content[0].text
except Exception as e:
    print(f"  FAILED: {e}")
    test3_ok = False

# Summary
print("=" * 60)
print("RESULTS")
print("=" * 60)
results = [
    ("Fast model text", test1_ok),
    ("Fast + system prompt", test2_ok),
    ("Smart model text", test3_ok),
]
all_ok = True
for name, ok in results:
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("All tests passed. Claude API is working correctly.")
    print("You can now run: python main.py")
else:
    print("Some tests FAILED. Check your API key and model names.")
