from urllib.parse import urlsplit, urljoin
from bs4 import BeautifulSoup

def normalize_url(url:str)->str:
  nrm_url = urlsplit(url)
  return nrm_url.netloc + nrm_url.path

def domain_url(url:str)->str:
  nrm_url = urlsplit(url)
  return nrm_url.netloc

def get_heading_from_html(html: str)-> str:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    h2 = soup.find("h2")
    if h1:
        return h1.get_text()
    elif h2:
        return h2.get_text()
    return ''

def get_urls_from_html(html: str, base_url: str)-> list:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue
        # Use urljoin to correctly resolve absolute, root-relative and relative URLs
        full = urljoin(base_url, href)
        if full.startswith("http"):
            urls.append(full)
    return urls

def get_images_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        full = urljoin(base_url, src)
        if full.startswith("http"):
            images.append(full)
    return images

def get_first_paragraph_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    p = soup.find("p")
    if p:
        return p.get_text()
    return ''

def extract_page_data(html, page_url):
    return {
        "url": page_url,
        "heading": get_heading_from_html(html),
        "images": get_images_from_html(html, page_url),
        "first_paragraph": get_first_paragraph_from_html(html),
        "image_urls": get_images_from_html(html, page_url),
        "page_urls": get_urls_from_html(html, page_url)
    }