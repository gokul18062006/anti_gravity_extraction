"""Test EasyOCR Tamil model loading and text extraction."""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import easyocr
    print("Initializing EasyOCR with Tamil...")
    reader = easyocr.Reader(['ta'], gpu=False, verbose=False)
    print("SUCCESS: EasyOCR Tamil model loaded!")
    
    # Test with any image in the directory
    for f in os.listdir('.'):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"\nTesting extraction on: {f}")
            results = reader.readtext(f)
            print(f"Detected {len(results)} text regions:")
            for (bbox, text, conf) in results[:10]:
                print(f"  [{conf:.2f}] {text}")
            break
    else:
        print("\nNo test images found. Model is ready for the app!")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
