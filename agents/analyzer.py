import os
import json
import aiohttp
import asyncio
from config.manager import ConfigManager
from pathlib import Path
from colorama import Fore, Style
from typing import List
import re
from datetime import datetime

class OpenRouterAnalyzer:
    def __init__(self, db):
        self.config = ConfigManager()
        self.db = db
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.analysis_folder = Path("analysis")
        self.analysis_folder.mkdir(exist_ok=True)
        self.user_prompt = None
        self.benchmark_data = {
            'start_time': None,
            'end_time': None,
            'epochs': [],
            'total_requests': 0,
            'total_tokens': 0
        }

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    async def analyze_urls(self, filtered_content: dict):
        try:
            self.benchmark_data['start_time'] = datetime.now()
            print(f"{Fore.CYAN}Performing comprehensive analysis...{Style.RESET_ALL}")
            
            if "final_summary" not in filtered_content:
                raise ValueError("Expected final summary data")
                
            if not self.user_prompt:
                raise ValueError("User prompt not set")
                
            summary = filtered_content["final_summary"]
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Extract and sort URLs
            urls = self._sort_urls_by_relevance(summary)
            most_relevant_url = urls[0] if urls else "No URL found"
            
            best_analysis = None
            best_score = 0.0
            
            # More aggressive adaptive epoch control
            max_epochs = 5
            min_epochs = 1  # Reduced from 2 to 1
            quality_threshold = 0.92  # Lowered from 0.95
            improvement_threshold = 0.03  # Lowered from 0.05
            high_confidence_threshold = 0.96  # New threshold for immediate exit
            
            for epoch in range(1, max_epochs + 1):
                epoch_start = datetime.now()
                print(f"\n{Fore.CYAN}Starting Epoch {epoch} analysis...{Style.RESET_ALL}")
                
                # More conservative temperature adjustment
                temperature = 0.1 + (epoch * 0.02)  # Reduced from 0.05
                temperature = min(temperature, 0.25)  # Lowered cap from 0.3
                
                prompt = f"""
                Analysis Epoch: {epoch}
                
                Directly answer this question: {self.user_prompt}
                Answer this question with enough detail to be useful, list key points and findings.
                
                Use this content as your source:
                {summary}
                
                Most relevant URL: {most_relevant_url}
                
                Requirements:
                - Verify information is current as of {current_date}
                - Be extremely concise
                - Focus only on the most critical information
                """
                
                payload = {
                    "model": self.config.ai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1500  # Reduced from 2000
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            analysis = data['choices'][0]['message']['content']
                            
                            # Calculate score with more stringent criteria
                            score = self._evaluate_analysis_quality(analysis, epoch)
                            
                            # Update benchmark data
                            self.benchmark_data['total_requests'] += 1
                            tokens = len(analysis.split())
                            self.benchmark_data['total_tokens'] += tokens
                            epoch_end = datetime.now()
                            epoch_duration = (epoch_end - epoch_start).total_seconds()
                            
                            self.benchmark_data['epochs'].append({
                                'epoch': epoch,
                                'duration': epoch_duration,
                                'score': score,
                                'tokens': tokens
                            })
                            
                            # New high confidence immediate exit
                            if score >= high_confidence_threshold:
                                print(f"{Fore.GREEN}Reached high confidence threshold ({score:.2f}), ending analysis immediately{Style.RESET_ALL}")
                                best_analysis = analysis
                                break
                                
                            # Adaptive exit conditions
                            if score >= quality_threshold:
                                print(f"{Fore.GREEN}Reached quality threshold ({score:.2f}), ending analysis{Style.RESET_ALL}")
                                best_analysis = analysis
                                break
                                
                            if best_analysis and (score - best_score) < improvement_threshold:
                                print(f"{Fore.YELLOW}Minimal improvement ({score - best_score:.2f}), ending analysis{Style.RESET_ALL}")
                                break
                                
                            if score > best_score:
                                best_analysis = analysis
                                best_score = score
                                print(f"{Fore.GREEN}New best analysis found! Score: {best_score:.2f}{Style.RESET_ALL}")
                                print(f"\n{Fore.CYAN}Best Analysis Preview:{Style.RESET_ALL}")
                                print(analysis[:800] + "...")
                            
                            # Early exit if we've passed min_epochs and have a good enough score
                            if epoch >= min_epochs and best_score >= 0.82:  # Lowered from 0.85
                                print(f"{Fore.GREEN}Good enough score after minimum epochs, ending analysis{Style.RESET_ALL}")
                                break
                                
                            print(f"Epoch {epoch} analysis preview:")
                            print(analysis[:800] + "...")
                            
                        else:
                            error = await response.text()
                            print(f"{Fore.RED}Epoch {epoch} failed: {error}{Style.RESET_ALL}")
            
            # Save benchmark report only at the end
            if best_analysis:
                print(f"\n{Fore.GREEN}Final analysis complete! Best score: {best_score:.2f}{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Final Analysis:{Style.RESET_ALL}")
                print(best_analysis)
                return best_analysis
            else:
                print(f"{Fore.RED}Failed to generate valid analysis{Style.RESET_ALL}")
                return None
        except Exception as e:
            print(f"{Fore.RED}Error during analysis: {e}{Style.RESET_ALL}")
            return None

    def _sort_urls_by_relevance(self, content: str) -> List[str]:
        # Extract URLs from content
        urls = re.findall(r'https?://[^\s]+', content)
        
        # Score URLs based on relevance factors
        scored_urls = []
        for url in urls:
            score = 0
            
            # Higher score for main domain mentions
            domain = re.sub(r'https?://(www\.)?', '', url)
            domain = re.sub(r'\/.*', '', domain)
            score += content.lower().count(domain) * 0.1
            
            # Higher score for exact URL mentions
            score += content.lower().count(url) * 0.2
            
            # Higher score for earlier mentions
            position = content.lower().find(url)
            if position != -1:
                score += (1 - (position / len(content))) * 0.3
                
            # Higher score for authoritative domains
            if any(auth in domain for auth in ['.gov', '.edu', '.org']):
                score += 0.2
                
            scored_urls.append((url, score))
        
        # Sort by score descending
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [url for url, score in scored_urls]

    def _get_content_for_url(self, url):
        # Implement content retrieval from your database or storage
        pass

    async def save_report(self, report):
        report_path = self.analysis_folder / "aggregated_analysis.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"{Fore.GREEN}Aggregated report saved to {report_path}{Style.RESET_ALL}")

    def _evaluate_analysis_quality(self, analysis: str, epoch: int) -> float:
        score = 0.0
        
        if analysis:
            score += 0.2
            
        current_year = datetime.now().year
        if str(current_year) in analysis:
            score += 0.1 + (0.02 * epoch)
            
        if "as of" in analysis.lower() or "current" in analysis.lower():
            score += 0.1
            
        structure_components = [
            "### Executive Summary",
            "### Key Findings",
            "### Detailed Analysis",
            "### Recommendations",
            "### Sources"
        ]
        for i, component in enumerate(structure_components):
            if component in analysis:
                score += 0.1 + (0.02 * epoch)
                
        if epoch == 1 and "facts" in analysis.lower():
            score += 0.1
        if epoch == 2 and "evidence" in analysis.lower():
            score += 0.1
        if epoch == 3 and "patterns" in analysis.lower():
            score += 0.1
        if epoch == 4 and "insights" in analysis.lower():
            score += 0.1
        if epoch == 5 and "recommendations" in analysis.lower():
            score += 0.1
            
        score += min(len(analysis) / (2000 + (epoch * 200)), 0.2)
        
        if "clearly" in analysis.lower() or "concisely" in analysis.lower():
            score += 0.05 * epoch
            
        depth_indicators = ["detailed", "in-depth", "comprehensive", "thorough"]
        for indicator in depth_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "specific" in analysis.lower() or "precise" in analysis.lower():
            score += 0.05 * epoch
            
        evidence_indicators = ["data", "statistics", "research", "study", "source"]
        for indicator in evidence_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "actionable" in analysis.lower() or "recommendation" in analysis.lower():
            score += 0.05 * epoch
            
        return min(score, 1.0)

async def main(urls):
    analyzer = OpenRouterAnalyzer()
    report = await analyzer.analyze_urls(urls)
    if report:
        await analyzer.save_report(report)

# Example usage
if __name__ == "__main__":
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5"
    ]
    asyncio.run(main(urls)) 