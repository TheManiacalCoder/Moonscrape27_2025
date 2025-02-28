import asyncio
from typing import List, Dict
from colorama import Fore, Style
from config.manager import ConfigManager
import aiohttp
from datetime import datetime

class LocalAgent:
    def __init__(self, db):
        self.db = db
        self.config = ConfigManager()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.user_prompt = None

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    async def _get_content_for_url(self, url: str) -> str:
        """Retrieve content for a specific URL from the database"""
        with self.db.conn:
            cursor = self.db.conn.cursor()
            cursor.execute('''SELECT content FROM seo_content 
                           JOIN urls ON seo_content.url_id = urls.id 
                           WHERE urls.url = ?''', (url,))
            result = cursor.fetchone()
            return result[0] if result else None

    async def _analyze_content(self, content: str) -> str:
        """Analyze content using the AI model"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        Analyze this content from the local knowledge base:
        {content}
        
        Respond to this question: {self.user_prompt}
        
        Format the response as:
        
        ### Key Findings
        - [Main point 1]
        - [Main point 2]
        - [Main point 3]
        
        ### Supporting Evidence
        - [Fact 1 with source]
        - [Fact 2 with source]
        
        ### Conclusion
        [Summary of findings]
        
        Requirements:
        - Verify information is current as of {current_date}
        - Be concise and factual
        - Use only information from the provided content
        """
        
        payload = {
            "model": self.config.ai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    print(f"{Fore.RED}Analysis error: {error}{Style.RESET_ALL}")
                    return None

    async def query_knowledge_base(self) -> str:
        """Query the local knowledge base and return a formatted response"""
        if not self.user_prompt:
            return "No question provided"
            
        # Get all URLs from database
        with self.db.conn:
            cursor = self.db.conn.cursor()
            cursor.execute('SELECT url FROM urls')
            urls = [row[0] for row in cursor.fetchall()]
        
        if not urls:
            return "No data available in local knowledge base"
        
        # Get content for all URLs
        tasks = [self._get_content_for_url(url) for url in urls]
        contents = await asyncio.gather(*tasks)
        
        # Filter out None values
        valid_contents = [c for c in contents if c is not None]
        
        if not valid_contents:
            return "No processable content found"
        
        # Analyze the combined content
        combined_content = "\n\n".join(valid_contents)
        analysis = await self._analyze_content(combined_content)
        
        if analysis:
            return f"""
{Fore.MAGENTA}Local Knowledge Base Response:{Style.RESET_ALL}
{Fore.CYAN}Question:{Style.RESET_ALL} {self.user_prompt}

{Fore.GREEN}Summary:{Style.RESET_ALL}
{analysis}

{Fore.YELLOW}Sources:{Style.RESET_ALL}
{", ".join(urls[:3])}... (total {len(urls)} sources)

{Fore.YELLOW}Note:{Style.RESET_ALL} This information is based on previously scraped content and may not be up-to-date.
"""
        return "Failed to generate response from local knowledge base" 