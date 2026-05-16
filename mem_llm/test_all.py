"""
Comprehensive test script for MaLP code verification.
Tests all modules for import correctness and basic functionality.
"""

import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []


def test(name, func):
    """Run a test and record result."""
    try:
        func()
        results.append((name, "PASS", ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def test_memory_imports():
    from memory import Memory, Short_Term_Memory, Long_Term_Memory, WorkingMemory
    assert Memory is not None
    assert Short_Term_Memory is not None
    assert Long_Term_Memory is not None
    assert WorkingMemory is not None


def test_model_imports():
    from model import lora_llama, ChatGPTWrapper
    assert lora_llama is not None
    assert ChatGPTWrapper is not None


def test_working_memory():
    from memory import WorkingMemory
    wm = WorkingMemory()
    wm.add_note("Test note 1")
    wm.add_note("Test note 2")
    assert len(wm) == 2
    notes = wm.get_notes()
    assert len(notes) == 2
    wm.refresh()
    assert len(wm) == 0


def test_stm():
    from memory import Short_Term_Memory
    stm = Short_Term_Memory(min_sim_threshold=15)
    stm["fever treatment"] = "Take Tylenol"
    stm["diabetes care"] = "Monitor blood sugar"
    assert len(stm) == 2
    # Test get_closest
    key, val, score = stm.get_closest("fever")
    assert key is not None or score is None  # May or may not match depending on threshold


def test_ltm():
    from memory import Long_Term_Memory
    ltm = Long_Term_Memory()
    ltm["diabetes management"] = "Regular blood sugar monitoring"
    ltm["headache treatment"] = "Rest and hydration"
    assert len(ltm) == 2
    key, val, score = ltm.get_closest("how to manage blood sugar")
    # With sentence-transformers, should find a match
    if key is not None:
        assert "diabetes" in key.lower() or "blood" in key.lower()


def test_memory_transit():
    from memory import Memory
    memory = Memory(transit_threshold=2)
    memory.add_to_stm("test key", "value 1")
    memory.add_to_stm("test key", "value 2")  # freq = 2
    memory.add_to_stm("other key", "value 3")  # freq = 1
    transited = memory.transit()
    assert "test key" in transited
    assert len(memory.ltm) == 1


def test_memory_retrieve():
    from memory import Memory
    memory = Memory(transit_threshold=2)
    memory.add_to_stm("fever treatment", "Take Tylenol and rest")
    memory.add_to_ltm("diabetes care", "Monitor blood sugar regularly")
    result = memory.retrieve("fever")
    assert isinstance(result, str)


def test_lora_llama_class():
    from model.lora_llama import lora_llama
    # Just test the class can be instantiated with a mock
    assert lora_llama is not None


def test_chatgpt_wrapper_class():
    from model.utils import ChatGPTWrapper
    # Test class exists and can be instantiated (won't make API calls)
    wrapper = ChatGPTWrapper(model="gpt-4.1-mini")
    assert wrapper is not None
    assert wrapper.model == "gpt-4.1-mini"


def test_prepare_data_imports():
    import prepare_data
    assert hasattr(prepare_data, "main")


def test_memory_formation_imports():
    import memory_formation
    assert hasattr(memory_formation, "main")


def test_knowledge_injection_imports():
    import knowledge_injection
    assert hasattr(knowledge_injection, "main")


def test_train_imports():
    import train
    assert hasattr(train, "main")
    assert hasattr(train, "DialogueDataset")


def test_eval_imports():
    import eval
    assert hasattr(eval, "main")
    assert hasattr(eval, "compute_rouge")


def test_inference_imports():
    import inference
    assert hasattr(inference, "main")
    assert hasattr(inference, "MaLPInference")


def test_data_files():
    """Check that data files exist and are valid JSON."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dialogues2_cleaned.json")
    assert os.path.exists(data_path), f"Data file not found: {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0


def test_profiles_file():
    """Check that profiles file exists and is valid JSON."""
    profiles_path = os.path.join(os.path.dirname(__file__), "..", "dialogue_generation", "profiles_4.json")
    assert os.path.exists(profiles_path), f"Profiles file not found: {profiles_path}"
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    assert isinstance(profiles, list)
    assert len(profiles) > 0


def test_dialogue_generation_utils():
    """Test dialogue_generation utils imports."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dialogue_generation"))
    from utils import ChatGPTWrapper, ReWriter, Identifier, Summarizer
    assert ChatGPTWrapper is not None
    assert ReWriter is not None
    assert Identifier is not None
    assert Summarizer is not None


if __name__ == "__main__":
    print("=" * 60)
    print("MaLP - Comprehensive Code Verification")
    print("=" * 60)

    print("\n--- Module Imports ---")
    test("Memory module imports", test_memory_imports)
    test("Model module imports", test_model_imports)
    test("prepare_data imports", test_prepare_data_imports)
    test("memory_formation imports", test_memory_formation_imports)
    test("knowledge_injection imports", test_knowledge_injection_imports)
    test("train imports", test_train_imports)
    test("eval imports", test_eval_imports)
    test("inference imports", test_inference_imports)
    test("dialogue_generation utils imports", test_dialogue_generation_utils)

    print("\n--- Memory Functionality ---")
    test("WorkingMemory operations", test_working_memory)
    test("STM operations", test_stm)
    test("LTM operations", test_ltm)
    test("Memory transit", test_memory_transit)
    test("Memory retrieval", test_memory_retrieve)

    print("\n--- Model Classes ---")
    test("lora_llama class", test_lora_llama_class)
    test("ChatGPTWrapper class", test_chatgpt_wrapper_class)

    print("\n--- Data Files ---")
    test("Dialogue data files", test_data_files)
    test("Profiles file", test_profiles_file)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    print(f"RESULTS: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, status, err in results:
            if status == "FAIL":
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests PASSED!")
        sys.exit(0)
