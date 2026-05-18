import json
import os

def write_json_report(page_data, file_name = "report.json") -> None:
    """Writes the given data to a JSON file.

    Converts page_data.values() to a list, sorts by the 'url' field, and writes JSON.
    """
    try:
        pages_list = list((page_data or {} ).values())
        sorted_pages = sorted(pages_list, key=lambda x: x.get("url", ""))
        __print_report(page_data)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sorted_pages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing JSON report: {e}")

def __print_report(page_data) -> None:
   # Print summary of collected data
    num_pages = len(page_data or {})
    print(f"Crawl complete. Pages collected: {num_pages}")
    for value in (page_data or {}).values():
        url = value.get("url") if isinstance(value, dict) else None
        heading = value.get("heading") if isinstance(value, dict) else None
        out_links = len(value.get("page_urls", [])) if isinstance(value, dict) else 0
        print(f"- {url} | heading: {heading!s} | outbound links: {out_links}")