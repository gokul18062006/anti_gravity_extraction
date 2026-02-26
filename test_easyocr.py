"""Test EasyOCR Tamil model loading."""
import traceback

try:
    import easyocr
    print("Initializing EasyOCR with Tamil...")
    reader = easyocr.Reader(['ta'], gpu=False, verbose=True)
    print("SUCCESS: EasyOCR Tamil model loaded!")
    print(f"Languages: {reader.lang_list}")
except Exception as e:
    print(f"\nERROR: {e}\n")
    traceback.print_exc()
