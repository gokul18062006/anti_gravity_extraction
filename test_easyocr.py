"""Test EasyOCR Tamil text extraction end-to-end."""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import easyocr
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['ta'], gpu=False, verbose=False)
    print("SUCCESS: EasyOCR Tamil model loaded!")
    print(f"Reader created with languages: ta")
    
    # Test with the user's uploaded image if it exists
    import os
    test_images = [f for f in os.listdir('.') if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if test_images:
        print(f"\nTesting extraction on: {test_images[0]}")
        results = reader.readtext(test_images[0])
        print(f"Detected {len(results)} text regions:")
        for (bbox, text, conf) in results:
            print(f"  [{conf:.2f}] {text}")
    else:
        print("\nNo test images found. The model is ready for use in the app!")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
