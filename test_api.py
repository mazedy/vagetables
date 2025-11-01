"""
Simple test script to verify the Image Classification API is working
"""

import requests
import sys

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running?")
        print("   Run: python main.py")
        return False

def test_classify(image_path):
    """Test image classification"""
    print(f"\n🖼️  Testing image classification with: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/classify", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification successful!")
            print(f"\n📊 Results:")
            print(f"   Top Prediction: {result['top_prediction']['label']}")
            print(f"   Confidence: {result['top_prediction']['confidence']}%")
            print(f"\n   All Predictions:")
            for pred in result['predictions']:
                print(f"   - {pred['label']}: {pred['confidence']}%")
            return True
        else:
            print(f"❌ Classification failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Image Classification API Test")
    print("=" * 60)
    
    # Test health endpoint
    if not test_health():
        sys.exit(1)
    
    # Test classification if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_classify(image_path)
    else:
        print("\n💡 To test image classification, run:")
        print("   python test_api.py path/to/your/image.jpg")
    
    print("\n" + "=" * 60)
    print("✨ Test complete!")
    print("=" * 60)
