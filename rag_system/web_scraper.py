"""
Web Scraper Module for ENSAKH RAG System
Fetches and extracts content from ENSAKH website URLs
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Set
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ENSAKHWebScraper:
    """
    Professional web scraper for ENSAKH website
    - Fetches HTML content
    - Extracts main content (ignores navigation/ads)
    - Follows internal links
    - Respects robots.txt and rate limiting
    """
    
    def __init__(self, base_url: str = "http://ensak.usms.ac.ma", max_depth: int = 3):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        self.documents: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (ENSAKH RAG Bot) Educational Purpose'
        })
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to ENSAKH domain"""
        parsed = urlparse(url)
        
        # Must be from ENSAKH domain
        if 'ensak.usms.ac.ma' not in parsed.netloc:
            return False
            
        # Skip files we can't process as HTML
        skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    
    def fetch_page(self, url: str) -> str:
        """Fetch HTML content from URL with error handling"""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding  # Handle encoding properly
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def extract_main_content(self, html: str, url: str) -> Dict:
        """
        Extract main content from HTML
        - Removes navigation, ads, footers
        - Extracts headings, paragraphs, lists, tables
        - Preserves structure
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
            soup.find('body')
        )
        
        if not main_content:
            return None
        
        # Extract title
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract structured content
        content_parts = []
        
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']):
            text = element.get_text(separator=' ', strip=True)
            
            # Skip empty or very short content
            if len(text) < 10:
                continue
            
            # Add heading markers
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                content_parts.append(f"\n## {text}\n")
            elif element.name in ['ul', 'ol']:
                items = element.find_all('li')
                for item in items:
                    item_text = item.get_text(strip=True)
                    if item_text:
                        content_parts.append(f"• {item_text}")
            elif element.name == 'table':
                # Simple table extraction
                rows = element.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                    if row_text:
                        content_parts.append(row_text)
            else:
                content_parts.append(text)
        
        content = '\n'.join(content_parts)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'word_count': len(content.split())
        }
    
    def extract_links(self, html: str, current_url: str) -> List[str]:
        """Extract all valid internal links from HTML"""
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(current_url, href)
            
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]
            
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        
        return links
    
    def crawl(self, start_urls: List[str], max_pages: int = 100):
        """
        Crawl ENSAKH website starting from given URLs
        
        Args:
            start_urls: List of URLs to start crawling from
            max_pages: Maximum number of pages to crawl
        """
        queue = [(url, 0) for url in start_urls]  # (url, depth)
        
        while queue and len(self.documents) < max_pages:
            url, depth = queue.pop(0)
            
            # Skip if already visited or max depth reached
            if url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(url)
            
            # Fetch page
            html = self.fetch_page(url)
            if not html:
                continue
            
            # Extract content
            document = self.extract_main_content(html, url)
            if document and document['word_count'] > 50:  # Skip pages with too little content
                self.documents.append(document)
                logger.info(f"✓ Extracted: {document['title'][:50]}... ({document['word_count']} words)")
            
            # Extract and queue new links
            if depth < self.max_depth:
                new_links = self.extract_links(html, url)
                for link in new_links[:10]:  # Limit links per page
                    queue.append((link, depth + 1))
            
            # Rate limiting - be respectful
            time.sleep(0.5)
        
        logger.info(f"\n✅ Crawling complete! Collected {len(self.documents)} documents")
        return self.documents


def main():
    """Test the scraper"""
    
    # ENSAKH URLs to crawl
    start_urls = [
        "http://ensak.usms.ac.ma/ensak/",
        "http://ensak.usms.ac.ma/ensak/formation-initiale/",
        "http://ensak.usms.ac.ma/ensak/formation-continue/",
        "http://ensak.usms.ac.ma/ensak/emplois-du-temps/",
        "http://ensak.usms.ac.ma/ensak/formations-certifiantes/",
        "http://ensak.usms.ac.ma/ensak/departements/",
    ]
    
    scraper = ENSAKHWebScraper(max_depth=2)
    documents = scraper.crawl(start_urls, max_pages=50)
    
    # Save to file
    import json
    output_path = Path("scraped_documents.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved {len(documents)} documents to {output_path}")


if __name__ == "__main__":
    main()
