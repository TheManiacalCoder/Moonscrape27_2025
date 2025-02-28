import re
import aiohttp
import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager

class ContentProcessor:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }

    async def extract_key_points(self, content: str) -> str:
        prompt = f"""
        Act as an SEO Content Specialist and analyze this content:
        {content}
        
        Extract and format the following:
        
        1. Main Topic (1 sentence)
        2. Primary Keywords (5-10)
        3. Secondary Keywords (10-15)
        4. Long-tail Opportunities (5-10)
        5. Content Structure Analysis
        6. SEO Optimization Suggestions
        7. Content Gaps
        8. Engagement Opportunities
        
        Format the output as follows:
        
        ### Main Topic
        [Main topic summary]
        
        ### Keywords
        - Primary: [comma separated]
        - Secondary: [comma separated]
        - Long-tail: [comma separated]
        
        ### Structure Analysis
        [Analysis of content structure]
        
        ### SEO Suggestions
        [Specific SEO improvements]
        
        ### Content Gaps
        [Missing content opportunities]
        
        ### Engagement
        [Engagement strategy suggestions]
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 10000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    print(f"{Fore.RED}AI extraction error: {error}{Style.RESET_ALL}")
                    return None

    async def process_urls(self, urls: List[str]) -> Dict[str, str]:
        processed_data = {}
        
        with self.db.conn:
            cursor = self.db.conn.cursor()
            
            for url in urls:
                try:
                    cursor.execute('''SELECT content FROM seo_content 
                                   JOIN urls ON seo_content.url_id = urls.id 
                                   WHERE urls.url = ?''', (url,))
                    result = cursor.fetchone()
                    
                    if result and result[0] and not result[0].startswith("Error:"):
                        key_points = await self.extract_key_points(result[0])
                        if key_points:
                            processed_data[url] = key_points
                            print(f"{Fore.GREEN}Processed: {url}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}Skipped (no key points): {url}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Skipped (error/invalid): {url}{Style.RESET_ALL}")
                        
                except Exception as e:
                    print(f"{Fore.RED}Error processing {url}: {str(e)}{Style.RESET_ALL}")
        
        return processed_data 