Week of 06/01/2026 - 06/05/2026

- Onboarding completed
- Python refresher course completed
- Pandas referesher course completed
- Environment partially set up
- Open Food Facts Dataset website visited and explored

Week of 06/08/2026

- Dataset downloaded
- Jupyter Notebook set up
- Dataset loaded in using Pandas
- Dataset explored/preliminary cleaning done
- Preliminary sample of 500 obtained\
First PR

Week of 06/15/2026

- Update segmenting logic
- Improve edge case detection
- Report statistics regarding data (edge case occurence proportions, segment split, etc)
- Begin Implementation A (Known List Brand Matching)

Notes:

- Segmenting:
    - 1: Brand perfectly retrievable using substring search on product name after normalization (removing special characters, removing extra spaces, and standardizing to lower case)
    - 2: Brand inferrable from product name such that there is some fuzz/noise (threshholds for acceptable fuzz prior to relegation into segment 3 vary by length, the longer the brand name the more fuzz/noise is acceptable and the threshhold is lower) and the fuzzy matching is not run on a common English word/common ingredient
    - 3: Brand not at all inferrable or present in product name

- Edge Cases:
    - Vague: A brand that is itself a common English word (top-1000 list) or a food ingredient/product name
    is too generic and risks being matched incorrectly.
    - non_latin: True when text contains a genuine non-Latin script or other non-ASCII that survives accent normalization.
    Accented Latin (e.g. é, ñ) folds to ASCII and is not flagged; plain punctuation is benign.
    - unique_chars: Flag non-Latin scripts or non-ASCII text after accent normalization in either field, or a space inside
    an individual brand name. Plain punctuation (dashes, apostrophes, commas, etc.) is benign.
    - brand_conflict: A multi-brand entry where every listed brand appears in the title as a whole word, so the correct
    brand is ambiguous to extract (e.g. the actual brand vs the retailer, especially if it is not listed first).
    - spurious_substring: Flag only entries where a naive (no word-boundary) substring matcher would extract the WRONG
    brand: the longest brand from the dataset's brand universe that occurs anywhere in the title is an
    unrelated brand sitting embedded inside a larger word (so word-boundary matching skips it), while
    none of the entry's own brands appear cleanly as a standalone token. These are exactly the cases
    where dropping word boundaries breaks extraction (e.g. brand "u" inside "uncle chipps").
    - noisy: An exceptionally long product title carries lots of extra descriptors/codes/numbers, making brand extractio noisy.
    - misspelled: the brand name is misspelled in the product title, for this reason we have to ensure fuzzy matching does not break down here (e.g. brand "apericube" appears as "aperocube")

Other Design Choices:

- Normalize: a function to perform various transformations (lower casing, accent strpping, etc) on product name and brand strings to standardize them to a common form for ease and precision.
- NaN brand entry rows are excluded from analysis because they provide no value for extracting due to lack of ground truth to compare to.
- Benchmark/Sample: a 750 entry sample has been taken and saved/frozen to an external file with just over 500 segment 1 & 2 entries and around 250 segment 3 entries for trials and tests. The reason for this split is that the original smaple was 500 entries with only segments 1 & 2, but due to the need of segment 3 for real world like analysis, 250 entries were added on from segment 3 seperately to, one, make sure we have a good amount of all segments to test given segment 3's sheer size, and two, to not change indices of entries we had saved earlier for manual analysis (specific instances of edge cases, for example).
- Statistics: Proportions of edge cases, segments, and etc. are displayed throughout the notebook for sanity checks and data analysis (seeing where, down the line, certain chracteristcs must be accounted for, like we had to do for the benchmark knowing that most entries were segment 3)

