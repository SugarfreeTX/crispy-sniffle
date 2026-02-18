# Instagram Content Creator - Usage Examples

# This file shows you how to use the content creator in different ways

from content_creator import main_content_pipeline

print("=== Content Creator Usage Examples ===\n")

# OPTION 1: AUTOMATIC POSTING (requires Instagram credentials in .env)
print("🤖 OPTION 1: Automatic Posting")
print("To automatically post to Instagram:")
print("1. Add your Instagram credentials to .env file:")
print("   INSTAGRAM_USERNAME=your_username") 
print("   INSTAGRAM_PASSWORD=your_password")
print("2. Run: main_content_pipeline(auto_post=True)")
print("3. The system will create content and post automatically\n")

# OPTION 2: MANUAL UPLOAD (saves files for you to upload)
print("📱 OPTION 2: Manual Upload (Safer & Recommended)")
print("To create files for manual upload:")
print("1. Run: main_content_pipeline(auto_post=False)")
print("2. System creates portfolio_update.jpg and instagram_caption.txt")
print("3. Upload the image to Instagram manually")
print("4. Copy/paste the caption from the text file\n")

print("=== RUNNING EXAMPLE (Manual Mode) ===")

# Run an example in manual mode (safer to start with)
try:
    result = main_content_pipeline(auto_post=False)
    if result:
        print("\n✅ Content creation successful!")
        print("📷 Check for portfolio_update.jpg")
        print("📝 Check for instagram_caption.txt")
    else:
        print("\n❌ Content creation failed")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n=== To Enable Auto-Posting ===")
print("1. Update your .env file with real Instagram credentials")
print("2. Change auto_post=False to auto_post=True")
print("3. Consider Instagram's rate limits and TOS")