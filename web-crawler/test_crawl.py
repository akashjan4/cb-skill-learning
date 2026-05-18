import unittest
from crawl import normalize_url, get_heading_from_html, get_urls_from_html, get_images_from_html, get_first_paragraph_from_html, extract_page_data

class TestCrawl(unittest.TestCase):
    def test_normalize_https_url(self):
        input_url = "https://www.fake.dev/blog/path"
        actual = normalize_url(input_url)
        expected = "www.fake.dev/blog/path"
        self.assertEqual(actual, expected)

    def test_normalize_url_query(self):
        input_url = "https://www.fake.dev/blog/path?query#fragment"
        actual = normalize_url(input_url)
        expected = "www.fake.dev/blog/path"
        self.assertEqual(actual, expected)

    def test_normalize_http_url(self):
      input_url = "http://www.fake.dev/blog/path"
      actual = normalize_url(input_url)
      expected = "www.fake.dev/blog/path"
      self.assertEqual(actual, expected)

class TestGetHeadingFromHtml(unittest.TestCase):
    def test_get_h1_from_html(self):
        html = "<html><body><h1>Test Heading 1</h1></body></html>"
        actual = get_heading_from_html(html)
        expected = "Test Heading 1"
        self.assertEqual(actual, expected)
      
    def test_get_h2_from_html(self):
      html = "<html><body><h2>Test Heading 2</h2></body></html>"
      actual = get_heading_from_html(html)
      expected = "Test Heading 2"
      self.assertEqual(actual, expected)

    def test_get_h1_only_from_html(self):
      html = "<html><body><h1>Test Heading 1</h1><h2>Test Heading 2</h2></body></html>"
      actual = get_heading_from_html(html)
      expected = "Test Heading 1"
      self.assertEqual(actual, expected)

    def test_get_empty_heading_from_html(self):
      html = "<html><body></body></html>"
      actual = get_heading_from_html(html)
      expected = ""
      self.assertEqual(actual, expected)

class TestGetUrlsFromHtml(unittest.TestCase):
    def test_get_urls_from_html(self):
        html = """
        <html>
            <body>
                <a href="/blog/post1">Post 1</a>
                <a href="https://www.example.com/blog/post2">Post 2</a>
            </body>
        </html>
        """
        base_url = "https://www.fake.dev"
        actual = get_urls_from_html(html, base_url)
        expected = [
            "https://www.fake.dev/blog/post1",
            "https://www.example.com/blog/post2"
        ]
        self.assertEqual(actual, expected)

    def test_get_no_urls_from_html(self):
      html = """
      <html>
          <body>
          </body>
      </html>
      """
      base_url = "https://www.fake.dev"
      actual = get_urls_from_html(html, base_url)
      expected = []
      self.assertEqual(actual, expected)

class TestGetImagesFromHtml(unittest.TestCase):
    def test_get_images_from_html(self):
        html = """
        <html>
            <body>
                <img src="/images/image1.jpg" />
                <img src="https://www.example.com/images/image2.jpg" />
            </body>
        </html>
        """
        base_url = "https://www.fake.dev"
        actual = get_images_from_html(html, base_url)
        expected = [
            "https://www.fake.dev/images/image1.jpg",
            "https://www.example.com/images/image2.jpg"
        ]
        self.assertEqual(actual, expected)

    def test_get_no_images_from_html(self):
      html = """
      <html>
          <body>
          </body>
      </html>
      """
      base_url = "https://www.fake.dev"
      actual = get_images_from_html(html, base_url)
      expected = []
      self.assertEqual(actual, expected)

class TestGetFirstParagraphFromHtml(unittest.TestCase):
    def test_get_first_paragraph_from_html(self):
        html = """
        <html>
            <body>
                <p>This is the first paragraph.</p>
                <p>This is the second paragraph.</p>
            </body>
        </html>
        """
        actual = get_first_paragraph_from_html(html)
        expected = "This is the first paragraph."
        self.assertEqual(actual, expected)

    def test_get_no_paragraph_from_html(self):
      html = """
      <html>
          <body>
          </body>
      </html>
      """
      actual = get_first_paragraph_from_html(html)
      expected = ""
      self.assertEqual(actual, expected)

class TestExtractPageData(unittest.TestCase):
  def test_extract_page_data(self):
      html = """
        <html>
          <body>
              <h1>Test Title</h1>
              <p>This is the first paragraph.</p>
              <a href="/link1">Link 1</a>
              <img src="/image1.jpg" alt="Image 1">
          </body>
        </html>
      """
      page_url = "https://crawler-test.com"
      actual = extract_page_data(html, page_url)
      expected = {
          "url": "https://crawler-test.com",
          "heading": "Test Title",
          "images": ["https://crawler-test.com/image1.jpg"],
          "first_paragraph": "This is the first paragraph.",
          "image_urls": ["https://crawler-test.com/image1.jpg"],
          "page_urls": ["https://crawler-test.com/link1"]
      }
      self.assertEqual(actual, expected)
      
  def test_extract_page_missing_data(self):
    html = """
      <html>
        <body>
        </body>
      </html>
    """
    page_url = "https://crawler-test.com"
    actual = extract_page_data(html, page_url)
    expected = {
        "url": "https://crawler-test.com",
        "heading": "",
        "images": [],
        "first_paragraph": "",
        "image_urls": [],
        "page_urls": []
    }
    self.assertEqual(actual, expected)
  
  def test_extract_page_some_data(self):
      html = """
        <html>
          <body>
              <h1>Test Title</h1>
              <a href="/link1">Link 1</a>
              <img src="/image1.jpg" alt="Image 1">
          </body>
        </html>
      """
      page_url = "https://crawler-test.com"
      actual = extract_page_data(html, page_url)
      expected = {
          "url": "https://crawler-test.com",
          "heading": "Test Title",
          "images": ["https://crawler-test.com/image1.jpg"],
          "first_paragraph": "",
          "image_urls": ["https://crawler-test.com/image1.jpg"],
          "page_urls": ["https://crawler-test.com/link1"]
      }
      self.assertEqual(actual, expected)
if __name__ == '__main__':
    unittest.main()