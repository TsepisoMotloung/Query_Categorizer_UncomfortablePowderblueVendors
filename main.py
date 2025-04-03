
from nltk.tokenize import punkt

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import nltk

def initialize_nltk():
    """Initialize NLTK by downloading required datasets"""
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)

    print(f"NLTK data will be stored in: {nltk_data_dir}")
    required_packages = ['punkt', 'wordnet', 'stopwords']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True, raise_on_error=True)
            package_dir = os.path.join(nltk_data_dir, package)
            print(f"Successfully downloaded {package} to {package_dir}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            raise

# Initialize NLTK packages
initialize_nltk()

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (uncomment the first time you run)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')


class AdvancedQueryCategorizer:

  def __init__(self):
    # Initialize NLP tools
    self.lemmatizer = WordNetLemmatizer()
    self.stopwords = set(nltk.corpus.stopwords.words('english'))

    # Define category patterns with synonyms and variations
    self.category_patterns = {
        "Date of Birth Issues": {
            'keywords': [
                'date of birth', 'dob', 'birth date', 'birthday',
                'birth not provided', 'birth certificate', 'date missing',
                'birth missing'
            ],
            'regexes': [
                r'(?:date|d\.?o\.?b\.?).*(?:birth|born).*(?:miss|not|incorrect)',
                r'(?:birth|born).*(?:date|d\.?o\.?b\.?).*(?:miss|not|incorrect)'
            ]
        },
        "ID Related": {
            'keywords': [
                'identity', 'id', 'identification', 'identity number',
                'id number', 'id document', 'identity document', 'id card',
                'id incorrect', 'identity incorrect', 'id not provided',
                'identity not provided', 'id verification', 'id validity',
                'identity validity', 'incomplete id'
            ],
            'regexes': [
                r'id(?:entity)?.*(?:num|no|card|doc).*(?:miss|not|incorrect|invalid|incomplete)',
                r'(?:miss|not|incorrect|invalid|incomplete).*id(?:entity)?'
            ]
        },
        "Beneficiary Issues": {
            'keywords': [
                'beneficiary', 'beneficiaries', 'nominee', 'nominees',
                'benefit recipient', 'benefit allocation',
                'benefit distribution', 'benefit assignment'
            ],
            'regexes': [
                r'beneficiar(?:y|ies).*(?:miss|not|incorrect|select|relationship)',
                r'(?:miss|not|incorrect).*beneficiar(?:y|ies)'
            ]
        },
        "Age Qualification": {
            'keywords': [
                'age', 'aged', 'aging', 'overage', 'over age', 'age limit',
                'age restriction', 'age requirement', 'age qualification',
                'qualify', 'qualification'
            ],
            'regexes': [
                r'age.*(?:limit|restrict|require|qualif|over)',
                r'(?:not|doesn\'?t).*qualify', r'over.*age', r'too.*old'
            ]
        },
        "Premium Issues": {
            'keywords': [
                'premium', 'premiums', 'payment', 'payments', 'fee', 'fees',
                'cost', 'monthly payment', 'amount', 'rate', 'pricing', 'price'
            ],
            'regexes': [
                r'premium.*(?:incorrect|wrong|invalid|miss|not)',
                r'rider.*premium', r'payment.*(?:incorrect|wrong|invalid)'
            ]
        },
        "Signature Issues": {
            'keywords': [
                'signature', 'sign', 'signing', 'signed', 'autograph',
                'endorsement', 'mark', 'signed document', 'authorize',
                'authorization'
            ],
            'regexes': [
                r'sign(?:ature)?.*(?:miss|not|incorrect|invalid|incomplete)',
                r'(?:miss|not).*sign(?:ature)?', r'no.*sign(?:ature)?'
            ]
        },
        "Bank Confirmation": {
            'keywords': [
                'bank', 'banking', 'account', 'confirmation', 'conformation',
                'verify', 'verification', 'validate', 'validation', 'proof',
                'statement', 'stop order'
            ],
            'regexes': [
                r'bank.*(?:confirm|proof|statement|miss|not|verification)',
                r'(?:miss|not).*bank.*(?:confirm|proof|statement)',
                r'stop\s*order'
            ]
        },
        "Territorial Limits": {
            'keywords': [
                'territorial', 'territory', 'region', 'location', 'residence',
                'residency', 'proof of residence', 'resident', 'address',
                'jurisdiction'
            ],
            'regexes': [
                r'territor(?:y|ial).*(?:limit|restrict)',
                r'proof.*(?:residence|address|location)', r'lesotho.*residence'
            ]
        },
        "Affordability": {
            'keywords': [
                'afford', 'affordability', 'budget', 'income', 'affordable',
                'financial', 'finance', 'capability', 'capacity', 'means'
            ],
            'regexes': [r'afford(?:ability)?', r'financial.*capacity']
        },
        "Rider Benefits": {
            'keywords': [
                'rider', 'riders', 'benefit', 'benefits', 'coverage', 'add-on',
                'additional', 'option', 'molebe', 'supplemental'
            ],
            'regexes': [
                r'rider.*(?:benefit|cover|option|molebe)',
                r'(?:miss|not|incorrect).*rider',
                r'rider.*(?:not|miss|incorrect)'
            ]
        },
        "Deduction Issues": {
            'keywords': [
                'deduction', 'deduct', 'deducted', 'withhold', 'withholding',
                'withdraw', 'withdrawal', 'payroll', 'employment number'
            ],
            'regexes': [
                r'deduct(?:ion)?.*(?:miss|not|incorrect|invalid)',
                r'employment.*number', r'payroll.*(?:miss|not|incorrect)'
            ]
        },
        "Declaration Issues": {
            'keywords': [
                'declaration', 'declare', 'statement', 'attestation',
                'checklist', 'check list', 'form completion', 'disclosure'
            ],
            'regexes': [
                r'declarat(?:ion)?.*(?:miss|not|incomplete|section)',
                r'check\s*list.*(?:miss|not|attached|incomplete)',
                r'(?:miss|not|incomplete).*declarat(?:ion)?'
            ]
        },
        "KYC Issues": {
            'keywords': [
                'kyc', 'k.y.c', 'know your customer', 'customer verification',
                'identity verification', 'customer due diligence', 'cdd'
            ],
            'regexes': [
                r'k\.?y\.?c\.?.*(?:miss|not|incomplete|form)',
                r'know.*customer.*(?:miss|not|incomplete)'
            ]
        }
    }

    # Create category exemplars for semantic matching
    self.category_exemplars = {
        category: ' '.join(info['keywords'])
        for category, info in self.category_patterns.items()
    }

    # Compile regex patterns
    for category, info in self.category_patterns.items():
      info['compiled_regexes'] = [
          re.compile(pattern, re.IGNORECASE) for pattern in info['regexes']
      ]

  def preprocess_text(self, text):
    """Preprocess text with tokenization, lemmatization and stopword removal"""
    if text is None:
      return ""

    text = str(text).lower().strip()
    tokens = word_tokenize(text)
    tokens = [
        self.lemmatizer.lemmatize(token) for token in tokens
        if token.isalpha() and token not in self.stopwords
    ]
    return ' '.join(tokens)

  def categorize_query(self, query, verbose=False):
    """
        Advanced query categorization using multiple techniques:
        1. Rule-based categorization using regex patterns
        2. Keyword matching with lemmatization
        3. Semantic similarity using TF-IDF and cosine similarity
        """
    if query is None or str(query).strip() == "":
      return "Other"

    raw_query = str(query).lower().strip()
    processed_query = self.preprocess_text(query)

    # Dictionary to store scores for each category
    category_scores = {
        category: 0
        for category in self.category_patterns.keys()
    }

    # 1. Rule-based matching
    for category, info in self.category_patterns.items():
      # Check regex patterns
      for regex in info['compiled_regexes']:
        if regex.search(raw_query):
          category_scores[category] += 3  # Higher weight for regex matches

      # Keyword matching
      for keyword in info['keywords']:
        if keyword in raw_query:
          category_scores[category] += 2

    # Calculate top two categories by rule-based scoring
    sorted_categories = sorted(category_scores.items(),
                               key=lambda x: x[1],
                               reverse=True)
    top_category = sorted_categories[0][0]
    top_score = sorted_categories[0][1]

    # If we have a clear winner with a good score, return it
    if top_score >= 3:
      if verbose:
        print(
            f"Categorized '{query}' as '{top_category}' with score {top_score}"
        )
      return top_category

    # 2. For ambiguous cases, use semantic similarity with TF-IDF
    if top_score < 3 or (len(sorted_categories) > 1
                         and sorted_categories[1][1] >= top_score - 1):
      # Prepare vectorizer
      tfidf_vectorizer = TfidfVectorizer()

      # Combine exemplars and query
      all_texts = list(self.category_exemplars.values())
      all_texts.append(processed_query)

      # Generate TF-IDF matrix
      tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

      # Calculate similarity between query and each category
      query_vector = tfidf_matrix[-1]
      semantic_scores = {}

      for i, category in enumerate(self.category_patterns.keys()):
        similarity = cosine_similarity(query_vector, tfidf_matrix[i])[0][0]
        semantic_scores[category] = similarity

      # Find best semantic match
      best_semantic_category = max(semantic_scores.items(), key=lambda x: x[1])

      if best_semantic_category[1] > 0.2:  # Threshold for semantic similarity
        if verbose:
          print(
              f"Semantically categorized '{query}' as '{best_semantic_category[0]}' with similarity {best_semantic_category[1]}"
          )
        return best_semantic_category[0]

    # 3. Special case handling for common patterns
    if "premium" in raw_query and any(term in raw_query
                                      for term in ["incorrect", "wrong"]):
      return "Premium Issues"

    if "rider" in raw_query and "premium" in raw_query:
      return "Premium Issues"

    if "identity" in raw_query or " id " in raw_query or "identity number" in raw_query:
      return "ID Related"

    if "age" in raw_query and any(term in raw_query
                                  for term in ["over", "not qualify"]):
      return "Age Qualification"

    if verbose and top_score > 0:
      print(f"Low confidence categorization of '{query}' as '{top_category}'")

    # Return best guess or "Other" if scores are too low
    return top_category if top_score > 0 else "Other"


class AdvancedQueryCategorizationTool:

  def __init__(self, master):
    self.master = master
    self.categorizer = AdvancedQueryCategorizer()
    self.setup_ui(master)

  def setup_ui(self, master):
    """Set up the enhanced UI"""
    master.title("Advanced Query Categorization Tool")
    master.geometry("700x600")

    # Use a modern theme if available
    style = ttk.Style()
    if "clam" in style.theme_names():
      style.theme_use("clam")

    # Create notebook for tabs
    self.notebook = ttk.Notebook(master)
    self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Main tab
    self.main_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.main_tab, text="Main")

    # Statistics tab
    self.stats_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.stats_tab, text="Statistics")

    # Testing tab
    self.test_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.test_tab, text="Test Categorization")

    # Setup main tab
    self.setup_main_tab(self.main_tab)

    # Setup statistics tab
    self.setup_stats_tab(self.stats_tab)

    # Setup testing tab
    self.setup_test_tab(self.test_tab)

    # Track directories and results
    self.input_directory = None
    self.output_directory = None
    self.last_categorization_results = None

  def setup_main_tab(self, tab):
    """Setup the main tab with input/output selection and process button"""
    # Input Directory Section
    input_frame = ttk.LabelFrame(tab, text="Input Settings")
    input_frame.pack(fill=tk.X, padx=10, pady=10)

    ttk.Label(input_frame,
              text="Select Input Directory with Query Files:").pack(pady=(10,
                                                                          5))

    self.input_button = ttk.Button(input_frame,
                                   text="Choose Input Directory",
                                   command=self.select_input_directory)
    self.input_button.pack(pady=5)

    self.input_path_display = ttk.Label(input_frame,
                                        text="",
                                        font=("Arial", 9, "italic"))
    self.input_path_display.pack(pady=5)

    # Output Directory Section
    output_frame = ttk.LabelFrame(tab, text="Output Settings")
    output_frame.pack(fill=tk.X, padx=10, pady=10)

    ttk.Label(output_frame,
              text="Select Output Directory for Categorized Files:").pack(
                  pady=(10, 5))

    self.output_button = ttk.Button(output_frame,
                                    text="Choose Output Directory",
                                    command=self.select_output_directory)
    self.output_button.pack(pady=5)

    self.output_path_display = ttk.Label(output_frame,
                                         text="",
                                         font=("Arial", 9, "italic"))
    self.output_path_display.pack(pady=5)

    # Advanced options frame
    options_frame = ttk.LabelFrame(tab, text="Options")
    options_frame.pack(fill=tk.X, padx=10, pady=10)

    # Verbose logging option
    self.verbose_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options_frame,
                    text="Enable detailed logging",
                    variable=self.verbose_var).pack(pady=5,
                                                    padx=10,
                                                    anchor=tk.W)

    # Test on sample option
    self.test_sample_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(options_frame,
                    text="Run accuracy test after processing",
                    variable=self.test_sample_var).pack(pady=5,
                                                        padx=10,
                                                        anchor=tk.W)

    # Process Button in its own frame
    process_frame = ttk.Frame(tab)
    process_frame.pack(fill=tk.X, padx=10, pady=20)

    self.process_button = ttk.Button(process_frame,
                                     text="Categorize Queries",
                                     command=self.process_files,
                                     state=tk.DISABLED)
    self.process_button.pack(pady=10, fill=tk.X)

    # Progress bar
    self.progress_var = tk.DoubleVar()
    self.progress_bar = ttk.Progressbar(process_frame,
                                        variable=self.progress_var,
                                        maximum=100)
    self.progress_bar.pack(fill=tk.X, pady=5)

    # Status Display
    self.status_label = ttk.Label(process_frame,
                                  text="Ready",
                                  font=("Arial", 9))
    self.status_label.pack(pady=5)

  def setup_stats_tab(self, tab):
    """Setup statistics display tab"""
    # Statistics frame
    stats_frame = ttk.Frame(tab)
    stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create a Text widget with scrollbar for statistics
    self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=20, width=60)
    scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
    self.stats_text.configure(yscrollcommand=scrollbar.set)

    self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Default text
    self.stats_text.insert(
        tk.END, "Statistics will appear here after processing files.")
    self.stats_text.config(state=tk.DISABLED)

  def setup_test_tab(self, tab):
    """Setup testing tab for manual categorization testing"""
    # Test frame
    test_frame = ttk.Frame(tab)
    test_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    ttk.Label(test_frame,
              text="Enter a query to test categorization:").pack(pady=(10, 5))

    # Query entry
    self.test_query_entry = ttk.Entry(test_frame, width=60)
    self.test_query_entry.pack(pady=10, fill=tk.X)

    # Test button
    test_button = ttk.Button(test_frame,
                             text="Test Categorization",
                             command=self.test_categorization)
    test_button.pack(pady=10)

    # Result frame
    result_frame = ttk.LabelFrame(test_frame, text="Results")
    result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # Result text
    self.test_result_text = tk.Text(result_frame, wrap=tk.WORD, height=15)
    test_scrollbar = ttk.Scrollbar(result_frame,
                                   command=self.test_result_text.yview)
    self.test_result_text.configure(yscrollcommand=test_scrollbar.set)

    self.test_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Batch test section
    batch_frame = ttk.LabelFrame(test_frame, text="Batch Testing")
    batch_frame.pack(fill=tk.X, pady=10)

    batch_button = ttk.Button(batch_frame,
                              text="Run Batch Tests",
                              command=self.run_batch_tests)
    batch_button.pack(pady=10)

  def select_input_directory(self):
    """Handle input directory selection"""
    self.input_directory = filedialog.askdirectory(
        title="Select Directory with Query Files")
    if self.input_directory:
      self.input_path_display.config(text=self.input_directory)
      self.check_process_readiness()

  def select_output_directory(self):
    """Handle output directory selection"""
    self.output_directory = filedialog.askdirectory(
        title="Select Output Directory for Categorized Files")
    if self.output_directory:
      self.output_path_display.config(text=self.output_directory)
      self.check_process_readiness()

  def check_process_readiness(self):
    """Enable process button if both directories are selected"""
    if self.input_directory and self.output_directory:
      self.process_button.config(state=tk.NORMAL)

  def process_files(self):
    """Process all Excel files in the input directory"""
    try:
      # Reset progress and status
      self.progress_var.set(0)
      self.status_label.config(text="Processing...", foreground="blue")
      self.master.update_idletasks()

      # Find Excel files
      excel_files = [
          f for f in os.listdir(self.input_directory)
          if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')
      ]

      if not excel_files:
        messagebox.showwarning(
            "No Files", "No Excel files found in the selected directory.")
        self.status_label.config(text="No files found", foreground="orange")
        return

      # Initialize counters and storage for statistics
      processed_count = 0
      total_files = len(excel_files)
      category_counts = {}
      all_processed_data = []

      # Process each file
      for i, file in enumerate(excel_files):
        try:
          file_path = os.path.join(self.input_directory, file)
          df = pd.read_excel(file_path)

          # Check if 'Query' column exists
          if 'Query' not in df.columns:
            messagebox.showwarning("Column Missing",
                                   f"No 'Query' column in {file}. Skipping.")
            continue

          # Store original categories if they exist
          has_original_categories = 'Category' in df.columns
          if has_original_categories:
            df['Original_Category'] = df['Category']

          # Categorize
          verbose = self.verbose_var.get()
          df['Category'] = df['Query'].apply(
              lambda q: self.categorizer.categorize_query(q, verbose))

          # Track categorization statistics
          for category in df['Category'].unique():
            count = len(df[df['Category'] == category])
            if category in category_counts:
              category_counts[category] += count
            else:
              category_counts[category] = count

          # Save to output directory
          output_file = os.path.join(
              self.output_directory,
              f"Categorized_{os.path.splitext(file)[0]}.csv")
          df.to_csv(output_file, index=False)

          # Add to all processed data for analysis
          all_processed_data.append(df)

          processed_count += 1

          # Update progress bar
          self.progress_var.set((i + 1) / total_files * 100)
          self.master.update_idletasks()

        except Exception as file_error:
          messagebox.showerror("File Processing Error",
                               f"Error processing {file}: {str(file_error)}")

      # Combine all processed data for analysis
      if all_processed_data:
        combined_df = pd.concat(all_processed_data, ignore_index=True)
        self.last_categorization_results = combined_df

        # Update statistics display
        self.update_statistics(category_counts, combined_df)

        # Test accuracy if original categories exist and option is selected
        if has_original_categories and self.test_sample_var.get():
          self.test_accuracy(combined_df)

      # Final status update
      self.status_label.config(
          text=f"Processed {processed_count} files successfully!",
          foreground="green")
      messagebox.showinfo(
          "Complete",
          f"Processed {processed_count} files.\nCheck output directory and Statistics tab."
      )

    except Exception as e:
      messagebox.showerror("Error", str(e))
      self.status_label.config(text="Processing failed", foreground="red")

  def update_statistics(self, category_counts, combined_df):
    """Update statistics display with categorization results"""
    self.stats_text.config(state=tk.NORMAL)
    self.stats_text.delete(1.0, tk.END)

    # Write header
    self.stats_text.insert(tk.END, "CATEGORIZATION STATISTICS\n", "header")
    self.stats_text.insert(tk.END, "=" * 40 + "\n\n")

    # Write category counts
    self.stats_text.insert(tk.END, "CATEGORY DISTRIBUTION:\n", "subheader")
    total_queries = sum(category_counts.values())

    for category, count in sorted(category_counts.items(),
                                  key=lambda x: x[1],
                                  reverse=True):
      percentage = (count / total_queries) * 100 if total_queries > 0 else 0
      self.stats_text.insert(tk.END,
                             f"{category}: {count} ({percentage:.1f}%)\n")

    self.stats_text.insert(tk.END, f"\nTotal Queries: {total_queries}\n\n")

    # If we have original categories, show comparison
    if 'Original_Category' in combined_df.columns:
      self.stats_text.insert(tk.END, "COMPARISON WITH ORIGINAL CATEGORIES:\n",
                             "subheader")

      # Calculate accuracy
      matches = sum(
          combined_df['Original_Category'] == combined_df['Category'])
      accuracy = (matches /
                  len(combined_df)) * 100 if len(combined_df) > 0 else 0

      self.stats_text.insert(tk.END, f"Overall Accuracy: {accuracy:.2f}%\n")
      self.stats_text.insert(
          tk.END, f"Matching Categories: {matches}/{len(combined_df)}\n\n")

      # Show per-category accuracy
      self.stats_text.insert(tk.END, "PER-CATEGORY ACCURACY:\n")
      for category in combined_df['Original_Category'].unique():
        category_df = combined_df[combined_df['Original_Category'] == category]
        category_matches = sum(
            category_df['Original_Category'] == category_df['Category'])
        category_accuracy = (category_matches / len(category_df)) * 100 if len(
            category_df) > 0 else 0

        self.stats_text.insert(
            tk.END, f"{category}: {category_accuracy:.2f}% "
            f"({category_matches}/{len(category_df)})\n")

    # Apply text tags
    self.stats_text.tag_configure("header", font=("Arial", 12, "bold"))
    self.stats_text.tag_configure("subheader", font=("Arial", 10, "bold"))

    self.stats_text.config(state=tk.DISABLED)

  def test_categorization(self):
    """Test a single query categorization"""
    query = self.test_query_entry.get().strip()
    if not query:
      messagebox.showwarning("Empty Query", "Please enter a query to test.")
      return

    # Clear previous results
    self.test_result_text.delete(1.0, tk.END)

    # Run categorization with verbose mode
    category = self.categorizer.categorize_query(query, verbose=True)

    # Display result
    self.test_result_text.insert(tk.END, f"Query: {query}\n\n")
    self.test_result_text.insert(tk.END, f"Categorized as: {category}\n\n")

    # Show score breakdown for each category
    self.test_result_text.insert(tk.END, "Category Matching Details:\n",
                                 "header")

    # Test against each category pattern
    for cat, info in self.categorizer.category_patterns.items():
      self.test_result_text.insert(tk.END, f"\n{cat}:\n", "subheader")

      # Check regex patterns
      regex_matches = []
      for i, regex in enumerate(info['compiled_regexes']):
        if regex.search(query.lower()):
          regex_matches.append(f"Pattern {i+1}: {info['regexes'][i]}")

      if regex_matches:
        self.test_result_text.insert(tk.END, "  Matching patterns:\n")
        for match in regex_matches:
          self.test_result_text.insert(tk.END, f"  - {match}\n")
      else:
        self.test_result_text.insert(tk.END, "  No matching patterns\n")

      # Check keywords
      keyword_matches = []
      for keyword in info['keywords']:
        if keyword in query.lower():
          keyword_matches.append(keyword)

      if keyword_matches:
        self.test_result_text.insert(tk.END, "  Matching keywords:\n")
        for keyword in keyword_matches:
          self.test_result_text.insert(tk.END, f"  - {keyword}\n")
      else:
        self.test_result_text.insert(tk.END, "  No matching keywords\n")

    # Apply text tags
    self.test_result_text.tag_configure("header", font=("Arial", 10, "bold"))
    self.test_result_text.tag_configure("subheader", font=("Arial", 9, "bold"))

  def run_batch_tests(self):
    """Run a set of predefined tests to validate categorization"""
    test_cases = [
        ("Identity number not provided", "ID Related"),
        ("Identity number not provided for spouse", "ID Related"),
        ("Identity number incorrect - spouse", "ID Related"),
        ("Premium incorrect", "Premium Issues"),
        ("RIDER-MOLEBE PREMIUM IS INCORRECT", "Premium Issues"),
        ("Rider Premium incorrect for parent", "Premium Issues"),
        ("Premium incorrect - spouse riders incorrect because of age",
         "Premium Issues"), ("Beneficiary not selected", "Beneficiary Issues"),
        ("RELATIONSHIP-RELATIONSHIP OF PREMIUM PAYER AND BENEFICIARY NOT PROVIDED",
         "Beneficiary Issues"),
        ("Date of Birth not provided for spouse", "Date of Birth Issues"),
        ("Date of Birth not provided for beneficiary", "Date of Birth Issues"),
        ("AGE-OTHER DEPENDANT IS OVERAGE", "Age Qualification"),
        ("Age -Client does not qualify for cover", "Age Qualification"),
        ("PARENT - OVER AGE", "Age Qualification"),
        ("POLICY HOLDER - OVER AGE", "Age Qualification"),
        ("DECLARATION CHECK LIST NOT ATTACHED", "Declaration Issues"),
        ("Declaration Section - Not Completely Filled", "Declaration Issues"),
        ("Declaration Section - No Agent Signature", "Signature Issues"),
        ("Signature not provided", "Signature Issues"),
        ("Client signature missing", "Signature Issues"),
        ("Bank confirmation/Stop order not attached", "Bank Confirmation"),
        ("Bank Account details not provided", "Bank Confirmation"),
        ("Proof of residence not provided", "Territorial Limits"),
        ("Client staying in Lesotho - requires proof of residence",
         "Territorial Limits"),
        ("Employment number not provided", "Deduction Issues"),
        ("PAYROLL-EMPLOYMENT NUMBER NOT PROVIDED", "Deduction Issues"),
        ("KYC form incomplete", "KYC Issues"),
        ("Customer due diligence not completed", "KYC Issues")
    ]

    # Clear previous results
    self.test_result_text.delete(1.0, tk.END)

    # Run tests
    results = []
    correct = 0

    for query, expected in test_cases:
      actual = self.categorizer.categorize_query(query)
      is_match = actual == expected
      if is_match:
        correct += 1
      results.append((query, expected, actual, is_match))

    # Display results
    accuracy = (correct / len(test_cases)) * 100 if test_cases else 0

    self.test_result_text.insert(tk.END, "BATCH TEST RESULTS\n", "header")
    self.test_result_text.insert(
        tk.END, f"Accuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})\n\n")

    # Display individual results
    for query, expected, actual, is_match in results:
      result_marker = "✓" if is_match else "✗"
      result_color = "green" if is_match else "red"

      self.test_result_text.insert(tk.END, f"{result_marker} ",
                                   f"result_{result_color}")
      self.test_result_text.insert(tk.END, f"Query: {query}\n")
      self.test_result_text.insert(tk.END, f"   Expected: {expected}\n")

      if is_match:
        self.test_result_text.insert(tk.END, f"   Actual: {actual}\n\n")
      else:
        self.test_result_text.insert(tk.END, "   Actual: ", "normal")
        self.test_result_text.insert(tk.END, f"{actual}\n\n", "incorrect")

    # Apply text tags
    self.test_result_text.tag_configure("header", font=("Arial", 11, "bold"))
    self.test_result_text.tag_configure("result_green",
                                        foreground="green",
                                        font=("Arial", 9, "bold"))
    self.test_result_text.tag_configure("result_red",
                                        foreground="red",
                                        font=("Arial", 9, "bold"))
    self.test_result_text.tag_configure("incorrect", foreground="red")
    self.test_result_text.tag_configure("normal", foreground="black")

  def test_accuracy(self, df):
    """Test accuracy against original categories if available"""
    if 'Original_Category' not in df.columns:
      return

    # Calculate overall accuracy
    matches = sum(df['Original_Category'] == df['Category'])
    total = len(df)
    accuracy = (matches / total) * 100 if total > 0 else 0

    # Create confusion matrix
    original_categories = sorted(df['Original_Category'].unique())
    predicted_categories = sorted(df['Category'].unique())
    all_categories = sorted(
        list(set(original_categories) | set(predicted_categories)))

    confusion = pd.DataFrame(0, index=all_categories, columns=all_categories)

    for i, row in df.iterrows():
      original = row['Original_Category']
      predicted = row['Category']
      confusion.loc[original, predicted] += 1

    # Display results in a separate window
    result_window = tk.Toplevel(self.master)
    result_window.title("Accuracy Test Results")
    result_window.geometry("600x500")

    # Add overall results
    frame = ttk.Frame(result_window, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Accuracy Test Results",
              font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(
        frame,
        text=f"Overall Accuracy: {accuracy:.2f}% ({matches}/{total})").pack(
            pady=5)

    # Add per-category stats
    ttk.Label(frame, text="Per-Category Stats:",
              font=("Arial", 12, "bold")).pack(pady=(15, 5), anchor=tk.W)

    stats_frame = ttk.Frame(frame)
    stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    # Create scrolled text for results
    results_text = tk.Text(stats_frame, wrap=tk.WORD, height=15)
    scrollbar = ttk.Scrollbar(stats_frame, command=results_text.yview)
    results_text.configure(yscrollcommand=scrollbar.set)

    results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Calculate per-category stats
    for category in original_categories:
      category_df = df[df['Original_Category'] == category]
      category_matches = sum(
          category_df['Original_Category'] == category_df['Category'])
      category_total = len(category_df)
      category_accuracy = (category_matches /
                           category_total) * 100 if category_total > 0 else 0

      results_text.insert(tk.END, f"{category}:\n")
      results_text.insert(
          tk.END,
          f"  Accuracy: {category_accuracy:.2f}% ({category_matches}/{category_total})\n"
      )

      # Show misclassifications
      if category_matches < category_total:
        misclassified = category_df[category_df['Original_Category'] !=
                                    category_df['Category']]
        results_text.insert(tk.END, "  Misclassified as:\n")

        for wrong_cat, count in misclassified['Category'].value_counts().items(
        ):
          results_text.insert(tk.END, f"    - {wrong_cat}: {count}\n")

      results_text.insert(tk.END, "\n")

    results_text.config(state=tk.DISABLED)

    # Add a close button
    ttk.Button(frame, text="Close",
               command=result_window.destroy).pack(pady=10)

  def run(self):
    """Start the application main loop"""
    self.master.mainloop()


def main():
  root = tk.Tk()
  app = AdvancedQueryCategorizationTool(root)
  app.run()


if __name__ == "__main__":
  main()