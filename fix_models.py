# fix_models.py
"""
This script fixes the version mismatch issues by:
1. Loading your old pickled models
2. Re-saving them properly with current versions
"""

import joblib
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("CREDIT DEFAULT MODEL FIX UTILITY")
print("=" * 60)
print("\nStarting model fix process...\n")

# Step 1: Load the old XGBoost model (with warnings suppressed)
print("📦 Loading old XGBoost model...")
old_xgb_model = None
if os.path.exists('credit_default_model.pkl'):
    try:
        old_xgb_model = joblib.load('credit_default_model.pkl')
        print("✓ Old XGBoost model loaded from credit_default_model.pkl")
    except Exception as e:
        print(f"✗ Error loading XGBoost model: {e}")
else:
    print("✗ credit_default_model.pkl not found")

# Step 2: Load the old scaler
print("\n📦 Loading old StandardScaler...")
old_scaler = None
if os.path.exists('scaler.pkl'):
    try:
        old_scaler = joblib.load('scaler.pkl')
        print("✓ Old scaler loaded from scaler.pkl")
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
else:
    print("✗ scaler.pkl not found")

# Step 3: Save XGBoost model using native format (recommended)
print("\n" + "=" * 60)
print("SAVING MODELS IN UPDATED FORMATS")
print("=" * 60)

if old_xgb_model is not None:
    print("\n💾 Saving XGBoost model in native formats...")
    try:
        # Save as JSON (human-readable, version-stable)
        old_xgb_model.save_model('xgboost_model.json')
        print("✓ XGBoost model saved as 'xgboost_model.json'")
        
        # Save as UBJ (binary format, more compact)
        old_xgb_model.save_model('xgboost_model.ubj')
        print("✓ XGBoost model saved as 'xgboost_model.ubj'")
        
        print("\n  ℹ️  These formats are version-stable and won't cause warnings!")
    except Exception as e:
        print(f"✗ Error saving XGBoost model: {e}")
else:
    print("\n⚠️  Skipping XGBoost model save (model not loaded)")

# Step 4: Re-save scaler with current sklearn version
if old_scaler is not None:
    print("\n💾 Re-saving StandardScaler with current sklearn version...")
    try:
        joblib.dump(old_scaler, 'scaler_new.pkl')
        print("✓ Scaler saved as 'scaler_new.pkl'")
        print("  ℹ️  This eliminates version mismatch warnings!")
    except Exception as e:
        print(f"✗ Error saving scaler: {e}")
else:
    print("\n⚠️  Skipping scaler save (scaler not loaded)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

files_created = []
if os.path.exists('xgboost_model.json'):
    files_created.append('xgboost_model.json')
if os.path.exists('xgboost_model.ubj'):
    files_created.append('xgboost_model.ubj')
if os.path.exists('scaler_new.pkl'):
    files_created.append('scaler_new.pkl')

if files_created:
    print("\n✅ Successfully created the following files:")
    for file in files_created:
        size = os.path.getsize(file) / 1024  # KB
        print(f"   • {file} ({size:.1f} KB)")
    
    print("\n📝 Next Steps:")
    print("   1. The updated app.py will automatically use these new files")
    print("   2. Run: streamlit run app.py")
    print("   3. Verify no version warnings appear")
    print("\n✨ Your app should now run without any warnings!")
else:
    print("\n❌ No files were created. Please check:")
    print("   • credit_default_model.pkl exists in current directory")
    print("   • scaler.pkl exists in current directory")
    print("   • You have write permissions in this directory")

print("\n" + "=" * 60)
print("TECHNICAL DETAILS")
print("=" * 60)
print("\nWhy these changes fix the warnings:")
print("  • XGBoost native format (.json/.ubj) is version-independent")
print("  • Re-saving scaler updates it to current sklearn version")
print("  • These formats are more reliable for production deployment")
print("\n" + "=" * 60)