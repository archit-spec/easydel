#!/usr/bin/env python3
"""
Hyperswitch Repository Token Analysis using Qwen3-30B-A3B Tokenizer
Analyzes token counts across multiple Hyperswitch repositories
"""

import os
import json
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import fnmatch

try:
    from transformers import AutoTokenizer
    import git
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Missing required packages. Install with: pip install transformers gitpython pandas matplotlib seaborn")
    print(f"Error: {e}")
    exit(1)

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: str
    language: str
    lines: int
    size_bytes: int
    token_count: int
    tokens_per_line: float
    tokens_per_byte: float

@dataclass
class RepoAnalysis:
    """Analysis results for an entire repository"""
    name: str
    url: str
    total_files: int
    total_lines: int
    total_size_bytes: int
    total_tokens: int
    avg_tokens_per_file: float
    avg_tokens_per_line: float
    avg_tokens_per_byte: float
    median_tokens_per_file: float
    median_tokens_per_line: float
    median_tokens_per_byte: float
    language_breakdown: Dict[str, int]
    file_analyses: List[FileAnalysis]

class HyperswitchTokenAnalyzer:
    """Main analyzer class for Hyperswitch repositories"""
    
    # Repository configurations
    REPOSITORIES = {
        "hyperswitch": {
            "url": "https://github.com/juspay/hyperswitch.git",
            "description": "Core payment switch (Rust)",
            "primary_language": "Rust"
        },
        "hyperswitch-web": {
            "url": "https://github.com/juspay/hyperswitch-web.git", 
            "description": "Web SDK (ReScript/TypeScript)",
            "primary_language": "ReScript/TypeScript"
        },
        "hyperswitch-client-core": {
            "url": "https://github.com/juspay/hyperswitch-client-core.git",
            "description": "Client core library (TypeScript)", 
            "primary_language": "TypeScript"
        },
        "hyperswitch-control-center": {
            "url": "https://github.com/juspay/hyperswitch-control-center.git",
            "description": "Control center dashboard (React/TypeScript)",
            "primary_language": "React/TypeScript"
        }
    }
    
    # File extensions to analyze
    FILE_EXTENSIONS = {
        '.rs': 'Rust',
        '.ts': 'TypeScript', 
        '.tsx': 'TypeScript JSX',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript JSX',
        '.res': 'ReScript',
        '.resi': 'ReScript Interface',
        '.py': 'Python',
        '.go': 'Go',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C Header',
        '.hpp': 'C++ Header',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.html': 'HTML',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.toml': 'TOML',
        '.md': 'Markdown',
        '.sql': 'SQL'
    }
    
    # Files/directories to ignore
    IGNORE_PATTERNS = [
        '*.git*', '*/node_modules/*', '*/target/*', '*/dist/*', '*/build/*',
        '*/coverage/*', '*/.next/*', '*/.cache/*', '*/vendor/*',
        '*.lock', '*.log', '*.tmp', '*.temp', '*.bak', '*.swp',
        '*/migrations/*', '*/docs/*', '*/examples/*', '*/tests/*'
    ]
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B"):
        """Initialize the analyzer with the specified tokenizer"""
        print(f"🚀 Initializing Hyperswitch Token Analyzer...")
        print(f"📝 Loading tokenizer: {model_name}")
        
        try:
            # Try to load Qwen3-30B-A3B tokenizer, fallback to Qwen2.5-32B
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model_name = model_name
            print(f"✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load {model_name}, trying fallback...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
                self.model_name = "Qwen/Qwen2.5-32B-Instruct"
                print(f"✅ Fallback tokenizer loaded")
            except Exception as e2:
                print(f"❌ Failed to load tokenizer: {e2}")
                raise
        
        self.base_dir = Path("./hyperswitch_repos")
        self.results_dir = Path("./analysis_results")
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def clone_repositories(self) -> None:
        """Clone all Hyperswitch repositories"""
        print("\n📦 Cloning repositories...")
        
        for repo_name, repo_info in self.REPOSITORIES.items():
            repo_path = self.base_dir / repo_name
            
            if repo_path.exists():
                print(f"📁 {repo_name} already exists, pulling latest...")
                try:
                    repo = git.Repo(repo_path)
                    repo.remotes.origin.pull()
                except Exception as e:
                    print(f"⚠️  Failed to pull {repo_name}: {e}")
            else:
                print(f"📥 Cloning {repo_name}...")
                try:
                    git.Repo.clone_from(repo_info["url"], repo_path)
                    print(f"✅ {repo_name} cloned successfully")
                except Exception as e:
                    print(f"❌ Failed to clone {repo_name}: {e}")
                    
    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on patterns"""
        file_str = str(file_path)
        return any(fnmatch.fnmatch(file_str, pattern) for pattern in self.IGNORE_PATTERNS)
        
    def get_language_from_extension(self, file_path: Path) -> Optional[str]:
        """Get programming language from file extension"""
        return self.FILE_EXTENSIONS.get(file_path.suffix.lower())
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            print(f"⚠️  Error tokenizing text: {e}")
            return 0
            
    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file and return token statistics"""
        if self.should_ignore_file(file_path) or not file_path.is_file():
            return None
            
        language = self.get_language_from_extension(file_path)
        if not language:
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = len(content.splitlines())
            size_bytes = len(content.encode('utf-8'))
            token_count = self.count_tokens(content)
            
            return FileAnalysis(
                path=str(file_path),
                language=language,
                lines=lines,
                size_bytes=size_bytes,
                token_count=token_count,
                tokens_per_line=token_count / max(lines, 1),
                tokens_per_byte=token_count / max(size_bytes, 1)
            )
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
            return None
            
    def analyze_repository(self, repo_name: str) -> Optional[RepoAnalysis]:
        """Analyze an entire repository"""
        repo_path = self.base_dir / repo_name
        if not repo_path.exists():
            print(f"❌ Repository {repo_name} not found")
            return None
            
        print(f"🔍 Analyzing {repo_name}...")
        
        file_analyses = []
        language_stats = defaultdict(int)
        
        # Walk through all files
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                analysis = self.analyze_file(file_path)
                if analysis:
                    file_analyses.append(analysis)
                    language_stats[analysis.language] += analysis.token_count
                    
        if not file_analyses:
            print(f"⚠️  No analyzable files found in {repo_name}")
            return None
            
        # Calculate aggregate statistics
        total_files = len(file_analyses)
        total_lines = sum(fa.lines for fa in file_analyses)
        total_size_bytes = sum(fa.size_bytes for fa in file_analyses)
        total_tokens = sum(fa.token_count for fa in file_analyses)

        # Calculate medians
        token_counts = [fa.token_count for fa in file_analyses]
        tokens_per_line_list = [fa.tokens_per_line for fa in file_analyses]
        tokens_per_byte_list = [fa.tokens_per_byte for fa in file_analyses]

        median_tokens_per_file = statistics.median(token_counts) if token_counts else 0
        median_tokens_per_line = statistics.median(tokens_per_line_list) if tokens_per_line_list else 0
        median_tokens_per_byte = statistics.median(tokens_per_byte_list) if tokens_per_byte_list else 0

        return RepoAnalysis(
            name=repo_name,
            url=self.REPOSITORIES[repo_name]["url"],
            total_files=total_files,
            total_lines=total_lines,
            total_size_bytes=total_size_bytes,
            total_tokens=total_tokens,
            avg_tokens_per_file=total_tokens / max(total_files, 1),
            avg_tokens_per_line=total_tokens / max(total_lines, 1),
            avg_tokens_per_byte=total_tokens / max(total_size_bytes, 1),
            median_tokens_per_file=median_tokens_per_file,
            median_tokens_per_line=median_tokens_per_line,
            median_tokens_per_byte=median_tokens_per_byte,
            language_breakdown=dict(language_stats),
            file_analyses=file_analyses
        )
        
    def analyze_all_repositories(self) -> List[RepoAnalysis]:
        """Analyze all Hyperswitch repositories"""
        print(f"\n🔬 Starting comprehensive analysis using {self.model_name}...")
        
        analyses = []
        for repo_name in self.REPOSITORIES.keys():
            analysis = self.analyze_repository(repo_name)
            if analysis:
                analyses.append(analysis)
                
        return analyses
        
    def generate_summary_report(self, analyses: List[RepoAnalysis]) -> None:
        """Generate and display summary report"""
        if not analyses:
            print("❌ No analysis data available")
            return
            
        print("\n" + "="*80)
        print("📊 HYPERSWITCH TOKEN ANALYSIS SUMMARY")
        print("="*80)
        print(f"🤖 Tokenizer: {self.model_name}")
        print(f"📁 Repositories analyzed: {len(analyses)}")
        
        # Overall statistics
        total_files = sum(a.total_files for a in analyses)
        total_lines = sum(a.total_lines for a in analyses)
        total_tokens = sum(a.total_tokens for a in analyses)
        total_size_mb = sum(a.total_size_bytes for a in analyses) / (1024 * 1024)
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Files: {total_files:,}")
        print(f"   Total Lines: {total_lines:,}")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Total Size: {total_size_mb:.1f} MB")
        print(f"   Avg Tokens/Line: {total_tokens/max(total_lines,1):.2f}")
        
        # Per-repository breakdown
        print(f"\n📦 PER-REPOSITORY BREAKDOWN:")
        print("-" * 80)
        
        for analysis in sorted(analyses, key=lambda x: x.total_tokens, reverse=True):
            print(f"\n🔹 {analysis.name.upper()}")
            print(f"   Description: {self.REPOSITORIES[analysis.name]['description']}")
            print(f"   Files: {analysis.total_files:,}")
            print(f"   Lines: {analysis.total_lines:,}")
            print(f"   Tokens: {analysis.total_tokens:,}")
            print(f"   Size: {analysis.total_size_bytes/(1024*1024):.1f} MB")
            print(f"   Avg Tokens/File: {analysis.avg_tokens_per_file:.0f}")
            print(f"   Avg Tokens/Line: {analysis.avg_tokens_per_line:.2f}")
            print(f"   Median Tokens/File: {analysis.median_tokens_per_file:.0f}")
            print(f"   Median Tokens/Line: {analysis.median_tokens_per_line:.2f}")

            # Top languages
            top_langs = sorted(analysis.language_breakdown.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top Languages: {', '.join(f'{lang}({tokens:,})' for lang, tokens in top_langs)}")
            
        # Language analysis across all repos
        print(f"\n🌍 LANGUAGE ANALYSIS (ALL REPOSITORIES):")
        print("-" * 50)
        
        all_languages = defaultdict(int)
        for analysis in analyses:
            for lang, tokens in analysis.language_breakdown.items():
                all_languages[lang] += tokens
                
        for lang, tokens in sorted(all_languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (tokens / total_tokens) * 100
            print(f"   {lang:20}: {tokens:>10,} tokens ({percentage:5.1f}%)")
            
    def save_detailed_results(self, analyses: List[RepoAnalysis]) -> None:
        """Save detailed results to files"""
        print(f"\n💾 Saving detailed results to {self.results_dir}...")
        
        # Save JSON summary
        summary_data = {
            "tokenizer": self.model_name,
            "total_repositories": len(analyses),
            "analyses": [asdict(analysis) for analysis in analyses]
        }
        
        with open(self.results_dir / "token_analysis.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        # Save CSV for each repository
        for analysis in analyses:
            df = pd.DataFrame([asdict(fa) for fa in analysis.file_analyses])
            df.to_csv(self.results_dir / f"{analysis.name}_files.csv", index=False)
            
        # Save overall summary CSV
        summary_df = pd.DataFrame([
            {
                "repository": a.name,
                "description": self.REPOSITORIES[a.name]["description"],
                "total_files": a.total_files,
                "total_lines": a.total_lines,
                "total_tokens": a.total_tokens,
                "total_size_mb": a.total_size_bytes / (1024 * 1024),
                "avg_tokens_per_file": a.avg_tokens_per_file,
                "avg_tokens_per_line": a.avg_tokens_per_line,
                "median_tokens_per_file": a.median_tokens_per_file,
                "median_tokens_per_line": a.median_tokens_per_line,
                "primary_language": max(a.language_breakdown.items(), key=lambda x: x[1])[0] if a.language_breakdown else "Unknown"
            }
            for a in analyses
        ])
        summary_df.to_csv(self.results_dir / "repository_summary.csv", index=False)
        
        print(f"✅ Results saved to {self.results_dir}")
        
    def create_visualizations(self, analyses: List[RepoAnalysis]) -> None:
        """Create visualization charts"""
        print(f"\n📊 Creating visualizations...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Repository comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Hyperswitch Token Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        
        repos = [a.name for a in analyses]
        tokens = [a.total_tokens for a in analyses]
        files = [a.total_files for a in analyses]
        lines = [a.total_lines for a in analyses]
        
        # Token count by repository
        ax1.bar(repos, tokens, color='skyblue', alpha=0.8)
        ax1.set_title('Total Tokens by Repository')
        ax1.set_ylabel('Tokens')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(tokens):
            ax1.text(i, v + max(tokens)*0.01, f'{v:,}', ha='center', va='bottom')
            
        # Files by repository
        ax2.bar(repos, files, color='lightcoral', alpha=0.8)
        ax2.set_title('Total Files by Repository') 
        ax2.set_ylabel('Files')
        ax2.tick_params(axis='x', rotation=45)
        
        # Lines by repository
        ax3.bar(repos, lines, color='lightgreen', alpha=0.8)
        ax3.set_title('Total Lines by Repository')
        ax3.set_ylabel('Lines')
        ax3.tick_params(axis='x', rotation=45)
        
        # Tokens per line
        tokens_per_line = [a.avg_tokens_per_line for a in analyses]
        ax4.bar(repos, tokens_per_line, color='gold', alpha=0.8)
        ax4.set_title('Average Tokens per Line')
        ax4.set_ylabel('Tokens/Line')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'repository_comparison.png', dpi=300, bbox_inches='tight')

        # 2. Average vs Median Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(repos))
        width = 0.35

        avg_tokens_per_file = [a.avg_tokens_per_file for a in analyses]
        median_tokens_per_file = [a.median_tokens_per_file for a in analyses]

        ax.bar([i - width/2 for i in x], avg_tokens_per_file, width, label='Average Tokens/File', color='skyblue', alpha=0.8)
        ax.bar([i + width/2 for i in x], median_tokens_per_file, width, label='Median Tokens/File', color='orange', alpha=0.8)

        ax.set_xlabel('Repository')
        ax.set_ylabel('Tokens per File')
        ax.set_title('Average vs Median Tokens per File by Repository')
        ax.set_xticks(x)
        ax.set_xticklabels(repos, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'avg_median_comparison.png', dpi=300, bbox_inches='tight')

        # 3. Token Distribution Histogram (across all files)
        plt.figure(figsize=(14, 10))

        # Collect all file data
        all_token_counts = []
        all_tokens_per_line = []
        repo_labels = []

        for analysis in analyses:
            for fa in analysis.file_analyses:
                all_token_counts.append(fa.token_count)
                all_tokens_per_line.append(fa.tokens_per_line)
                repo_labels.append(analysis.name)

        # Create subplots for distributions
        plt.subplot(2, 2, 1)
        plt.hist(all_token_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Tokens per File')
        plt.ylabel('Frequency')
        plt.title('Distribution of Tokens per File (All Repositories)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.hist(all_tokens_per_line, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Tokens per Line')
        plt.ylabel('Frequency')
        plt.title('Distribution of Tokens per Line (All Repositories)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        # Box plot by repository
        repo_data = [[] for _ in analyses]
        for i, analysis in enumerate(analyses):
            for fa in analysis.file_analyses:
                repo_data[i].append(fa.token_count)

        plt.boxplot(repo_data, labels=repos)
        plt.xlabel('Repository')
        plt.ylabel('Tokens per File')
        plt.title('Token Distribution by Repository (Box Plot)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Scatter plot of file size vs tokens
        all_file_sizes = []
        all_file_tokens = []
        for analysis in analyses:
            for fa in analysis.file_analyses:
                all_file_sizes.append(fa.size_bytes / 1024)  # Convert to KB
                all_file_tokens.append(fa.token_count)

        plt.scatter(all_file_sizes, all_file_tokens, alpha=0.6, color='green', s=10)
        plt.xlabel('File Size (KB)')
        plt.ylabel('Tokens')
        plt.title('File Size vs Token Count')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'token_distributions.png', dpi=300, bbox_inches='tight')

        # 3.5. Per-File Token Histogram (X-axis: Token count, Y-axis: Number of files)
        plt.figure(figsize=(14, 8))

        # Create histogram of token counts per file
        plt.hist(all_token_counts, bins=50, alpha=0.7, color='purple', edgecolor='black', density=False)

        plt.xlabel('Number of Tokens per File')
        plt.ylabel('Number of Files')
        plt.title('Distribution of Token Counts Across All Files\n(X-axis: Tokens per File, Y-axis: Number of Files)')

        # Add statistics text
        mean_tokens = sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0
        median_tokens = statistics.median(all_token_counts) if all_token_counts else 0
        max_tokens = max(all_token_counts) if all_token_counts else 0

        stats_text = '.0f'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'per_file_token_histogram.png', dpi=300, bbox_inches='tight')

        # 4. Language distribution
        plt.figure(figsize=(12, 8))
        all_languages = defaultdict(int)
        for analysis in analyses:
            for lang, tokens in analysis.language_breakdown.items():
                all_languages[lang] += tokens

        # Top languages pie chart
        top_langs = dict(sorted(all_languages.items(), key=lambda x: x[1], reverse=True)[:8])
        others = sum(all_languages.values()) - sum(top_langs.values())
        if others > 0:
            top_langs['Others'] = others

        plt.pie(top_langs.values(), labels=top_langs.keys(), autopct='%1.1f%%', startangle=90)
        plt.title(f'Token Distribution by Language\n(Total: {sum(all_languages.values()):,} tokens)')
        plt.savefig(self.results_dir / 'language_distribution.png', dpi=300, bbox_inches='tight')

        # 5. Individual repository language distribution bar charts
        print(f"📊 Creating individual repository language distribution charts...")

        colors = plt.cm.tab10.colors  # Use a colormap for consistent colors

        for analysis in analyses:
            if not analysis.language_breakdown:
                continue

            # Sort languages by token count (descending)
            sorted_langs = sorted(analysis.language_breakdown.items(),
                                key=lambda x: x[1], reverse=True)

            languages = [lang for lang, _ in sorted_langs]
            token_counts = [count for _, count in sorted_langs]

            # Create figure
            plt.figure(figsize=(12, 8))

            # Create horizontal bar chart
            bars = plt.barh(languages, token_counts, color=colors[:len(languages)], alpha=0.8)

            plt.xlabel('Token Count')
            plt.ylabel('Programming Language')
            plt.title(f'Language Distribution in {analysis.name}\n({analysis.total_tokens:,} total tokens)')

            # Add value labels on bars
            for bar, count in zip(bars, token_counts):
                plt.text(bar.get_width() + max(token_counts) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{count:,}',
                        ha='left', va='center', fontweight='bold')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save individual chart
            filename = f"{analysis.name}_language_distribution.png"
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory

            print(f"  ✅ Saved {filename}")
    
            # 6. Individual repository token histograms
            print(f"📊 Creating individual repository token histograms...")
    
            for analysis in analyses:
                if not analysis.file_analyses:
                    continue
    
                # Get token counts for this repository
                repo_token_counts = [fa.token_count for fa in analysis.file_analyses]
    
                if not repo_token_counts:
                    continue
    
                plt.figure(figsize=(12, 8))
    
                # Create histogram
                plt.hist(repo_token_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
                plt.xlabel('Number of Tokens per File')
                plt.ylabel('Number of Files')
                plt.title(f'Token Distribution in {analysis.name}\n({analysis.total_files} files, {analysis.total_tokens:,} total tokens)')
    
                # Add statistics
                mean_tokens = sum(repo_token_counts) / len(repo_token_counts)
                median_tokens = statistics.median(repo_token_counts)
                max_tokens = max(repo_token_counts)
    
                stats_text = '.0f'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
    
                # Save individual repository histogram
                filename = f"{analysis.name}_token_histogram.png"
                plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
    
                print(f"  ✅ Saved {filename}")
    
            # 7. Individual language token histograms
            print(f"📊 Creating individual language token histograms...")
    
            # Collect all files by language
            language_files = defaultdict(list)
    
            for analysis in analyses:
                for fa in analysis.file_analyses:
                    language_files[fa.language].append(fa)
    
            # Create histogram for each language
            for language, files in language_files.items():
                if len(files) < 5:  # Skip languages with too few files
                    continue
    
                token_counts = [fa.token_count for fa in files]
                total_tokens = sum(token_counts)
                total_files = len(files)
    
                plt.figure(figsize=(12, 8))
    
                plt.hist(token_counts, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    
                plt.xlabel('Number of Tokens per File')
                plt.ylabel('Number of Files')
                plt.title(f'Token Distribution for {language} Files\n({total_files} files, {total_tokens:,} total tokens)')
    
                # Add statistics
                mean_tokens = sum(token_counts) / len(token_counts)
                median_tokens = statistics.median(token_counts)
                max_tokens = max(token_counts)
    
                stats_text = '.0f'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
    
                # Save individual language histogram
                safe_language = language.replace('/', '_').replace(' ', '_')
                filename = f"{safe_language}_token_histogram.png"
                plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
    
                print(f"  ✅ Saved {filename}")
    
            print(f"✅ All visualizations saved to {self.results_dir}")

def main():
    """Main execution function"""
    analyzer = HyperswitchTokenAnalyzer()
    
    try:
        # Clone repositories
        analyzer.clone_repositories()
        
        # Perform analysis
        analyses = analyzer.analyze_all_repositories()
        
        if analyses:
            # Generate reports
            analyzer.generate_summary_report(analyses)
            analyzer.save_detailed_results(analyses)
            analyzer.create_visualizations(analyses)
            
            print(f"\n🎉 Analysis complete! Check {analyzer.results_dir} for detailed results.")
        else:
            print("❌ No repositories could be analyzed")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()