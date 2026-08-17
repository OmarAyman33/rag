import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import wikipediaapi

INPUT_ROOT = "C:/Users/omara/Desktop/RAG/learning-rag/input"
START_TITLE = "Iron Man"
TARGET_COUNT = 1000
MAX_WORKERS = 10  # keep modest; Wikipedia's API rate-limits aggressive parallel clients

# Link namespaces to skip -- these aren't real articles
NAMESPACE_PREFIXES = (
    "Wikipedia:", "Help:", "Category:", "Template:", "Portal:", "File:",
    "Draft:", "Talk:", "User:", "User talk:", "Module:", "Special:",
    "MediaWiki:", "TimedText:", "Book:", "Template talk:", "Category talk:",
)

os.makedirs(INPUT_ROOT, exist_ok=True)

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="MyProjectName (merlin@example.com)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

# Shared state, all access guarded by `lock`
lock = threading.Lock()
visited = set()      # titles already queued/fetched, prevents duplicate work
write_count = 0       # articles successfully written so far


def sanitize_filename(title: str) -> str:
    """Turn a wiki title into a safe filename."""
    name = re.sub(r'[^\w\-. ]', "_", title)
    return name.strip()[:150]


def is_article_link(title: str) -> bool:
    """Filter out non-article namespaces (Category:, File:, etc.)."""
    return ":" not in title or not any(
        title.startswith(prefix) for prefix in NAMESPACE_PREFIXES
    )


def fetch_and_save(title: str):
    """Fetch a page, write it to disk, return (canonical_title, links) or None."""
    try:
        page = wiki_wiki.page(title)
        if not page.exists():
            return None

        text = page.text
        if not text:
            return None

        filename = sanitize_filename(page.title) + ".txt"
        filepath = os.path.join(INPUT_ROOT, filename)

        # Defensive dedup check in case two differently-cased titles resolve
        # to the same canonical page/filename
        if os.path.exists(filepath):
            return None

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        return page.title, list(page.links.keys())
    except Exception as e:
        print(f"Error fetching '{title}': {e}")
        return None


def main():
    global write_count

    frontier = deque([START_TITLE])
    visited.add(START_TITLE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while frontier and write_count < TARGET_COUNT:
            # Pull a batch off the frontier to fetch in parallel
            batch = []
            while frontier and len(batch) < MAX_WORKERS * 4:
                batch.append(frontier.popleft())

            futures = {executor.submit(fetch_and_save, title): title for title in batch}
            discovered_links = []

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                title, links = result
                with lock:
                    if write_count >= TARGET_COUNT:
                        continue
                    write_count += 1
                    print(f"[{write_count}/{TARGET_COUNT}] Saved: {title}")

                discovered_links.extend(links)

            # Enqueue newly discovered, unvisited article links for the next BFS level
            with lock:
                for link_title in discovered_links:
                    if write_count + len(frontier) >= TARGET_COUNT:
                        break
                    if link_title not in visited and is_article_link(link_title):
                        visited.add(link_title)
                        frontier.append(link_title)

    print(f"\nDone. {write_count} articles saved to {INPUT_ROOT}")


if __name__ == "__main__":
    main()