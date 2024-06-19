import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution = {}

    # for each page, first add probability of the next page being chosen at random over all the other pages
    for pg in corpus.keys():
        prob = (1 - damping_factor) / len(corpus.keys())
        prob_distribution.update({pg: prob})

    # for each link and the page that they go to, add probability of that link being chosen out of all the other pages
    for link in corpus[page]:
        prob_distribution[link] += damping_factor / len(corpus[page])

    # Exception for pages without links -> overrides previous calculations
    if corpus[page] == {}:
        for pg in corpus.keys():
            prob = 1 / len(corpus.keys())
            prob_distribution.update({pg: prob})
    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {pg: 0 for pg in list(corpus.keys())}
    prob_distribution = {}
    for i in range(n):
        if i == 0:
            page = random.choice(list(corpus.keys()))
        else:
            # choosing between choices with weighted probabilities
            page = random.choices(list(corpus.keys()), list(prob_distribution.values()))[0]
        pageranks[page] += 1/n
        prob_distribution = transition_model(corpus, page, damping_factor)
    return pageranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {}
    N = len(corpus.keys())

    # initial value for PageRanks
    for page in corpus.keys():
        prob = 1 / N
        pageranks.update({page: prob})

    # iterates until no PageRank value changes by more than 0.001
    while True:
        diff = 0
        for page_p in corpus.keys():
            new_prob = (1 - damping_factor) / N

            # looping through pages that have links going to page_p
            for page_i in corpus.keys():
                if corpus[page_i] == set():
                    new_prob += damping_factor * pageranks[page_i] / N
                if page_p in corpus[page_i]:
                    new_prob += damping_factor * pageranks[page_i] / len(corpus[page_i])
            diff = max(diff, abs(new_prob - pageranks[page_p]))
            pageranks[page_p] = new_prob
        if diff <= 0.001:
            break
    return pageranks


if __name__ == "__main__":
    main()
