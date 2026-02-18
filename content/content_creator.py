import json
import requests
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from instagrapi import Client
from PIL import Image, ImageDraw, ImageFont
import io

load_dotenv()

def generate_content_strategy(portfolio_data, trading_recommendation, api_key=os.getenv("GROK_API_KEY")):
    """
    Use Grok to generate engaging content ideas and captions based on portfolio performance and trading recommendations
    """
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Calculate portfolio metrics for content
    current_portfolio_value = portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0))
    cost_basis_total = portfolio_data.get('shares', 0) * portfolio_data.get('cost_basis', 0)
    performance_pct = ((current_portfolio_value - cost_basis_total) / cost_basis_total * 100) if cost_basis_total > 0 else 0
    
    prompt = f"""
    You are a social media content strategist for a successful trader/content creator. 
    
    Create engaging Instagram content that will get SAVES and SHARES in 2026 when content creators are rewarded for engagement.
    
    PORTFOLIO PERFORMANCE:
    - Current Portfolio Value: ${current_portfolio_value:,.2f}
    - Performance: {performance_pct:+.2f}%
    - Current Stock Price: ${portfolio_data.get('current_price', 0):.2f}
    - Shares Owned: {portfolio_data.get('shares', 0)}
    - Today's Recommendation: {trading_recommendation}
    
    MARKET DATA:
    - Recent Price Movement: {portfolio_data.get('price_change_pct', 0):+.2f}%
    - Volume: {portfolio_data.get('volume', 0):,}
    - Volatility: {portfolio_data.get('volatility', 0):.2f}
    
    Create content that focuses on:
    1. The STORY behind this trade/performance 
    2. Educational value people can SAVE for later
    3. Relatable content people want to SHARE
    
    Provide your response in this EXACT JSON format:
    {{
        "video_concept": "One engaging video idea that showcases the performance/decision",
        "hook_line": "Attention-grabbing first line for the video",
        "caption": "Full Instagram caption with storytelling and educational value",
        "hashtags": "List of trending finance hashtags for maximum reach",
        "call_to_action": "Specific CTA to encourage saves/shares",
        "content_type": "Type of video (e.g., 'results reveal', 'strategy breakdown', 'learning moment')"
    }}
    
    Make it personal, educational, and shareable. Focus on the trading results and decision-making process, NOT technical backend stuff.
    """
    
    body = {
        "model": "grok-4-1-fast-reasoning-latest", 
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=body)
        response_data = response.json()

        # print("API Response Status:", response.status_code)
        # print("API Response Body:", response_data)
        
        # Extract the content from Grok's response
        grok_content = response_data['choices'][0]['message']['content']
        
        # Try to parse as JSON, if it fails, return the raw content
        try:
            content_strategy = json.loads(grok_content)
            return content_strategy
        except json.JSONDecodeError:
            # If Grok doesn't return valid JSON, create a structured response
            return {
                "video_concept": "Portfolio performance analysis",
                "hook_line": "Here's what happened with my portfolio today...",
                "caption": grok_content,
                "hashtags": "#trading #stocks #investing #portfolio #finance",
                "call_to_action": "Save this for your trading journey!",
                "content_type": "results reveal"
            }
            
    except Exception as e:
        print(f"Error generating content strategy: {e}")
        return None

def create_performance_story(portfolio_data, previous_portfolio_data=None):
    """
    Create a narrative around portfolio performance changes
    """
    current_value = portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0))
    
    if previous_portfolio_data:
        prev_value = previous_portfolio_data.get('cash', 0) + (previous_portfolio_data.get('shares', 0) * previous_portfolio_data.get('previous_price', 0))
        daily_change = current_value - prev_value
        daily_change_pct = (daily_change / prev_value * 100) if prev_value > 0 else 0
        
        if daily_change > 0:
            story_type = "win"
            emotion = "excited"
        elif daily_change < 0:
            story_type = "learning_moment" 
            emotion = "reflective"
        else:
            story_type = "steady"
            emotion = "patient"
            
        return {
            "story_type": story_type,
            "emotion": emotion,
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "current_value": current_value
        }
    
    return {
        "story_type": "update",
        "emotion": "informative", 
        "current_value": current_value
    }

def generate_video_script(content_strategy, performance_story):
    """
    Generate a detailed video script based on content strategy and performance
    """
    script_sections = {
        "hook": content_strategy.get("hook_line", ""),
        "setup": f"Today's portfolio update - {performance_story['story_type']} story",
        "main_content": f"Here's what happened and why I'm {content_strategy.get('call_to_action', 'taking action')}",
        "educational_moment": "Key lesson you can apply to your own trading",
        "call_to_action": content_strategy.get("call_to_action", "")
    }
    
    return script_sections

def create_portfolio_image(portfolio_data, content_strategy):
    """
    Create a simple portfolio performance image for Instagram posting
    """
    # Create a simple image with portfolio stats
    img_width, img_height = 1080, 1080
    img = Image.new('RGB', (img_width, img_height), color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a nice font, fall back to default if not available
        title_font = ImageFont.truetype("arial.ttf", 80)
        subtitle_font = ImageFont.truetype("arial.ttf", 60) 
        body_font = ImageFont.truetype("arial.ttf", 40)
    except:
        # Fallback to default font if arial not found
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Calculate portfolio metrics
    current_portfolio_value = portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0))
    cost_basis_total = portfolio_data.get('shares', 0) * portfolio_data.get('cost_basis', 0)
    performance_pct = ((current_portfolio_value - cost_basis_total) / cost_basis_total * 100) if cost_basis_total > 0 else 0
    
    # Choose colors based on performance
    if performance_pct > 0:
        accent_color = '#00ff88'  # Green for gains
    elif performance_pct < 0:
        accent_color = '#ff4757'  # Red for losses
    else:
        accent_color = '#ffa502'  # Orange for flat
    
    # Add content to image
    y_position = 100
    
    # Title
    title = "Portfolio Update"
    draw.text((540, y_position), title, font=title_font, fill='white', anchor='mm')
    y_position += 150
    
    # Performance percentage
    perf_text = f"{performance_pct:+.2f}%"
    draw.text((540, y_position), perf_text, font=title_font, fill=accent_color, anchor='mm')
    y_position += 150
    
    # Portfolio value
    value_text = f"${current_portfolio_value:,.2f}"
    draw.text((540, y_position), value_text, font=subtitle_font, fill='white', anchor='mm')
    y_position += 100
    
    # Current price
    price_text = f"Price: ${portfolio_data.get('current_price', 0):.2f}"
    draw.text((540, y_position), price_text, font=body_font, fill='#cccccc', anchor='mm')
    y_position += 80
    
    # Shares
    shares_text = f"Shares: {portfolio_data.get('shares', 0)}"
    draw.text((540, y_position), shares_text, font=body_font, fill='#cccccc', anchor='mm')
    y_position += 120
    
    # Action
    hook_line = content_strategy.get('hook_line', 'Portfolio Update')
    # Wrap text if too long
    if len(hook_line) > 50:
        hook_line = hook_line[:47] + "..."
    draw.text((540, y_position), hook_line, font=body_font, fill=accent_color, anchor='mm')
    
    # Save image
    img_path = Path("portfolio_update.jpg")
    img.save(img_path, quality=95)
    return img_path

def setup_instagram_client():
    """
    Set up Instagram client with credentials from environment variables
    
    IMPORTANT: Add these to your .env file:
    INSTAGRAM_USERNAME=your_username
    INSTAGRAM_PASSWORD=your_password
    """
    username = os.getenv('INSTAGRAM_USERNAME')
    password = os.getenv('INSTAGRAM_PASSWORD')
    
    if not username or not password:
        print("⚠️  Instagram credentials not found!")
        print("Add these lines to your .env file:")
        print("INSTAGRAM_USERNAME=your_instagram_username")
        print("INSTAGRAM_PASSWORD=your_instagram_password")
        return None, "missing_credentials"
    
    try:
        cl = Client()
        cl.login(username, password)
        print(f"✅ Successfully logged into Instagram as @{username}")
        return cl, "success"
    except Exception as e:
        print(f"❌ Instagram login failed: {e}")
        return None, str(e)

def post_to_instagram(content_strategy, portfolio_data, auto_post=True):
    """
    Post content to Instagram automatically or save files for manual upload
    
    Args:
        auto_post (bool): If True, posts automatically. If False, saves files for manual upload.
    """
    
    if auto_post:
        # AUTOMATIC POSTING
        cl, status = setup_instagram_client()
        
        if cl is None:
            print("Switching to manual mode - saving files for you to upload manually...")
            return post_to_instagram(content_strategy, portfolio_data, auto_post=False)
        
        try:
            # Create portfolio image
            img_path = create_portfolio_image(portfolio_data, content_strategy)
            
            # Prepare caption with hashtags
            caption = content_strategy.get('caption', '')
            hashtags = content_strategy.get('hashtags', '')
            call_to_action = content_strategy.get('call_to_action', '')
            
            full_caption = f"{caption}\n\n{call_to_action}\n\n{hashtags}"
            
            # Post to Instagram
            media = cl.photo_upload(img_path, full_caption)
            
            print("🚀 POSTED TO INSTAGRAM!")
            print(f"Media ID: {media.pk}")
            print(f"Caption: {full_caption[:100]}...")
            
            # Clean up
            if os.path.exists(img_path):
                os.remove(img_path)
                
            return {"status": "posted", "media_id": media.pk}
            
        except Exception as e:
            print(f"❌ Posting failed: {e}")
            print("Saving files for manual upload instead...")
            return post_to_instagram(content_strategy, portfolio_data, auto_post=False)
    
    else:
        # MANUAL MODE - Save files for manual upload
        try:
            # Create portfolio image
            img_path = create_portfolio_image(portfolio_data, content_strategy)
            
            # Save caption to text file
            caption = content_strategy.get('caption', '')
            hashtags = content_strategy.get('hashtags', '')
            call_to_action = content_strategy.get('call_to_action', '')
            
            full_caption = f"{caption}\n\n{call_to_action}\n\n{hashtags}"
            
            caption_file = "instagram_caption.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(full_caption)
            
            print("💾 FILES SAVED FOR MANUAL UPLOAD:")
            print(f"📷 Image: {img_path}")
            print(f"📝 Caption: {caption_file}")
            print("\n=== READY FOR MANUAL UPLOAD ===")
            print("1. Open Instagram app or instagram.com")
            print(f"2. Upload the image: {img_path}")
            print(f"3. Copy the caption from: {caption_file}")
            print("4. Post to your story or feed!")
            print("===============================")
            
            return {
                "status": "saved_for_manual_upload", 
                "image_file": img_path,
                "caption_file": caption_file,
                "caption": full_caption
            }
            
        except Exception as e:
            print(f"❌ Error creating files: {e}")
            return {"status": "error", "message": str(e)}

def post_to_instagram_placeholder(content_strategy, video_path=None):
    """
    Placeholder function for Instagram posting integration
    This is where you'd integrate with instagrapi or Instagram Graph API
    """
    print("=== CONTENT READY FOR INSTAGRAM ===")
    print(f"Video Concept: {content_strategy.get('video_concept', '')}")
    print(f"Content Type: {content_strategy.get('content_type', '')}")
    print(f"\nCaption:\n{content_strategy.get('caption', '')}")
    print(f"\nHashtags: {content_strategy.get('hashtags', '')}")
    print(f"\nCall to Action: {content_strategy.get('call_to_action', '')}")
    print("=== READY TO POST ===")
    
    # TODO: Implement actual Instagram posting
    # from instagrapi import Client
    # cl = Client()
    # cl.login("username", "password") 
    # cl.video_upload(video_path, content_strategy.get('caption', ''))

def main_content_pipeline(auto_post=False):
    """
    Main function to run the complete content creation pipeline
    
    Args:
        auto_post (bool): If True, automatically posts to Instagram. If False, saves files for manual upload.
    """
    try:
        # Load current portfolio data
        with open('portfolio_state.json', 'r') as f:
            portfolio_data = json.load(f)
        
        # Get the latest trading recommendation (you'd call your grok trading function here)
        # For now, we'll simulate this
        trading_recommendation = "BUY"  # This would come from your trading analysis
        
        print(f"Creating content for {trading_recommendation} recommendation...")
        
        # Generate content strategy using Grok
        content_strategy = generate_content_strategy(portfolio_data, trading_recommendation)
        
        if content_strategy:
            # Create performance story
            performance_story = create_performance_story(portfolio_data)
            
            # Generate video script  
            video_script = generate_video_script(content_strategy, performance_story)
            
            # Display the content plan
            print("\n=== CONTENT CREATION COMPLETE ===")
            print(json.dumps(content_strategy, indent=2))
            print(f"\nPerformance Story: {performance_story}")
            print(f"\nVideo Script Sections: {video_script}")
            
            # Post to Instagram (automatic or manual)
            result = post_to_instagram(content_strategy, portfolio_data, auto_post=auto_post)
            
            return {
                "content_strategy": content_strategy,
                "performance_story": performance_story,
                "video_script": video_script,
                "instagram_result": result
            }
        else:
            print("Failed to generate content strategy")
            return None
            
    except Exception as e:
        print(f"Error in content pipeline: {e}")
        return None

if __name__ == "__main__":
    # You can change this to True for automatic posting
    # or keep False to save files for manual upload
    main_content_pipeline(auto_post=False)