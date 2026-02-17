import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def generate_video_prompts(portfolio_data, trading_recommendation, api_key=os.getenv("GROK_API_KEY")):
    """
    Use Grok to generate video concepts and detailed prompts for ComfyUI Desktop App
    """
    
    import requests
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Calculate portfolio metrics
    current_portfolio_value = portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0))
    cost_basis_total = portfolio_data.get('shares', 0) * portfolio_data.get('cost_basis', 0)
    performance_pct = ((current_portfolio_value - cost_basis_total) / cost_basis_total * 100) if cost_basis_total > 0 else 0
    
    prompt = f"""
    You are a video content strategist creating engaging trading content for Instagram Reels in 2026.
    
    Create a compelling video concept for ComfyUI desktop app based on this portfolio performance:
    
    PORTFOLIO DATA:
    - Portfolio Value: ${current_portfolio_value:,.2f}
    - Performance: {performance_pct:+.2f}%
    - Stock Price: ${portfolio_data.get('current_price', 0):.2f}
    - Shares: {portfolio_data.get('shares', 0)}
    - Recommendation: {trading_recommendation}
    - Price Movement: {portfolio_data.get('price_change_pct', 0):+.2f}%
    
    Create content that will get SAVES and SHARES. Focus on visual storytelling.
    
    Provide your response in this EXACT JSON format:
    {{
        "video_concept": "Engaging hook and video concept",
        "main_prompt": "Detailed positive prompt for ComfyUI (include portfolio performance, trading theme, professional style)",
        "negative_prompt": "Things to avoid in the video generation",
        "style_notes": "Visual style guidance (colors based on performance, mood, professional look)",
        "scene_descriptions": [
            "Scene 1: Opening hook visual description",
            "Scene 2: Portfolio reveal description", 
            "Scene 3: Educational moment description",
            "Scene 4: Call to action visual"
        ],
        "text_overlays": [
            "Key text to overlay on video (performance %, key insights, call to action)"
        ],
        "caption": "Instagram caption with storytelling",
        "hashtags": "Trending hashtags for maximum reach",
        "duration": "Recommended video length",
        "hook_line": "Opening line to grab attention",
        "call_to_action": "Specific CTA for engagement"
    }}
    
    Make prompts detailed for AI video generation. Include:
    - Professional trading aesthetic
    - Performance-based color scheme (green for gains, red for losses)
    - Clean, modern financial graphics
    - Engaging visual elements that encourage saves/shares
    
    Performance context: {performance_pct:+.2f}% with {trading_recommendation} recommendation.
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
        
        grok_content = response_data['choices'][0]['message']['content']
        
        try:
            video_strategy = json.loads(grok_content)
            return video_strategy
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "video_concept": "Portfolio performance reveal video",
                "main_prompt": f"professional trading portfolio review, {performance_pct:+.2f}% performance, modern financial graphics, clean aesthetic, high quality, detailed charts, trading dashboard",
                "negative_prompt": "blurry, low quality, unprofessional, cluttered, amateur",
                "style_notes": "Professional financial content with performance-based colors",
                "scene_descriptions": [
                    "Opening with portfolio dashboard showing current value",
                    "Animated chart revealing today's performance", 
                    "Clean infographic explaining the trading decision",
                    "Call to action with engagement prompt"
                ],
                "text_overlays": [f"Portfolio: {performance_pct:+.2f}%", f"Recommendation: {trading_recommendation}", "Save this strategy!"],
                "caption": grok_content,
                "hashtags": "#trading #stocks #portfolio #investing #finance",
                "duration": "15-30 seconds",
                "hook_line": f"Portfolio {performance_pct:+.2f}% today",
                "call_to_action": "Save for your trading journey!"
            }
            
    except Exception as e:
        print(f"Error generating video prompts: {e}")
        return None

def save_comfyui_prompts(video_strategy, portfolio_data):
    """
    Save detailed prompts and instructions for manual use in ComfyUI Desktop App
    """
    
    if not video_strategy:
        print("❌ No video strategy to save")
        return None
    
    try:
        # Calculate performance for file naming
        performance_pct = ((portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0))) - 
                          (portfolio_data.get('shares', 0) * portfolio_data.get('cost_basis', 0))) / \
                         (portfolio_data.get('shares', 0) * portfolio_data.get('cost_basis', 0)) * 100 if portfolio_data.get('shares', 0) > 0 else 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed prompt file for ComfyUI
        prompt_content = f"""=== PORTFOLIO VIDEO PROMPTS FOR COMFYUI ===
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Portfolio Performance: {performance_pct:+.2f}%

=== VIDEO CONCEPT ===
{video_strategy.get('video_concept', 'Portfolio performance video')}

=== MAIN POSITIVE PROMPT (Copy to ComfyUI) ===
{video_strategy.get('main_prompt', 'Professional portfolio trading content')}

=== NEGATIVE PROMPT (Copy to ComfyUI) ===  
{video_strategy.get('negative_prompt', 'blurry, low quality, unprofessional')}

=== STYLE NOTES ===
{video_strategy.get('style_notes', 'Professional financial aesthetic')}

=== SCENE DESCRIPTIONS ===
"""
        
        # Add scene descriptions
        scenes = video_strategy.get('scene_descriptions', [])
        for i, scene in enumerate(scenes, 1):
            prompt_content += f"Scene {i}: {scene}\n"
        
        prompt_content += f"""
=== TEXT OVERLAYS TO ADD ===
"""
        
        # Add text overlays
        text_overlays = video_strategy.get('text_overlays', [])
        for i, text in enumerate(text_overlays, 1):
            prompt_content += f"Overlay {i}: {text}\n"
        
        prompt_content += f"""
=== RECOMMENDED SETTINGS ===
- Video Format: Instagram Reels (9:16 aspect ratio, 1080x1920)
- Duration: {video_strategy.get('duration', '15-30 seconds')}
- Quality: High quality, professional finish
- Color theme: {"Green/success theme" if performance_pct > 0 else "Red/caution theme" if performance_pct < 0 else "Blue/neutral theme"}

=== PORTFOLIO DATA FOR REFERENCE ===
- Portfolio Value: ${portfolio_data.get('cash', 0) + (portfolio_data.get('shares', 0) * portfolio_data.get('current_price', 0)):,.2f}
- Performance: {performance_pct:+.2f}%
- Stock Price: ${portfolio_data.get('current_price', 0):.2f}
- Shares: {portfolio_data.get('shares', 0)}

=== INSTAGRAM CAPTION (Save separately) ===
{video_strategy.get('caption', 'Portfolio update content')}

{video_strategy.get('call_to_action', 'Save this for your trading journey!')}

{video_strategy.get('hashtags', '#trading #stocks #investing #finance')}
"""
        
        # Save main prompt file
        prompt_filename = f"comfyui_prompts_{timestamp}.txt"
        with open(prompt_filename, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        
        # Save Instagram caption separately
        caption_content = f"{video_strategy.get('caption', '')}\n\n{video_strategy.get('call_to_action', '')}\n\n{video_strategy.get('hashtags', '')}"
        caption_filename = f"instagram_video_caption_{timestamp}.txt"
        with open(caption_filename, 'w', encoding='utf-8') as f:
            f.write(caption_content)
        
        # Save quick reference JSON
        json_filename = f"video_strategy_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(video_strategy, f, indent=2)
        
        print("💾 FILES CREATED FOR COMFYUI:")
        print(f"📋 Main Prompts: {prompt_filename}")
        print(f"📝 Instagram Caption: {caption_filename}")  
        print(f"📄 JSON Reference: {json_filename}")
        print("\n=== MANUAL WORKFLOW ===")
        print("1. Open ComfyUI Desktop App")
        print(f"2. Copy positive prompt from: {prompt_filename}")
        print(f"3. Copy negative prompt from: {prompt_filename}")
        print("4. Set to Instagram Reels format (1080x1920)")
        print("5. Generate your video")
        print(f"6. Use caption from: {caption_filename} for Instagram")
        print("============================")
        
        return {
            "status": "prompts_saved",
            "prompt_file": prompt_filename,
            "caption_file": caption_filename,
            "json_file": json_filename,
            "video_strategy": video_strategy
        }
        
    except Exception as e:
        print(f"❌ Error saving prompts: {e}")
        return {"status": "error", "message": str(e)}

def main_video_pipeline():
    """
    Main function to generate ComfyUI prompts and instructions for manual use
    """
    try:
        # Load current portfolio data
        with open('portfolio_state.json', 'r') as f:
            portfolio_data = json.load(f)
        
        # Get trading recommendation (integrate with your existing system)
        trading_recommendation = "BUY"  # This would come from your trading analysis
        
        print("🎬 PORTFOLIO VIDEO PROMPT GENERATOR")
        print("=" * 45)
        print("Generating prompts for ComfyUI Desktop App...")
        
        # Generate video strategy with Grok
        video_strategy = generate_video_prompts(portfolio_data, trading_recommendation)
        
        if video_strategy:
            print("✅ Video strategy generated by Grok!")
            print(f"Concept: {video_strategy.get('video_concept', 'N/A')}")
            print(f"Duration: {video_strategy.get('duration', 'N/A')}")
            
            # Save files for manual use in ComfyUI
            result = save_comfyui_prompts(video_strategy, portfolio_data)
            
            return {
                "video_strategy": video_strategy,
                "save_result": result
            }
        else:
            print("❌ Failed to generate video strategy")
            return None
            
    except Exception as e:
        print(f"Error in video pipeline: {e}")
        return None

if __name__ == "__main__":
    print("🎥 Portfolio Video Prompt Generator")
    print("Creates prompts for ComfyUI Desktop App")
    print("=" * 50)
    
    main_video_pipeline()