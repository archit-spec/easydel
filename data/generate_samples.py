#!/usr/bin/env python3
import json
import os
import random
import string

def generate_random_string(length=10):
    words = ["payment", "gateway", "webhook", "transaction", "refund", "charge", "customer", "merchant", "api", "integration", "processor", "stripe", "paypal", "checkout", "authorization", "capture", "void", "settlement", "fraud", "compliance", "routing", "connector", "metadata", "callback", "encryption", "tokenization", "recurring", "subscription", "invoice", "dispute"]
    if length < 5:
        return random.choice(words)[:length]
    num_words = random.randint(1, 3)
    selected_words = random.choices(words, k=num_words)
    result = ' '.join(selected_words)
    if len(result) > length:
        result = result[:length]
    return result

def generate_commit():
    return {
        "hash": generate_random_string(40),
        "message": f"Fix {generate_random_string(20)} issue",
        "modifications": [
            {
                "filename": f"{generate_random_string(10)}.py",
                "change_type": random.choice(["A", "M", "D"]),
                "added": random.randint(0, 100),
                "removed": random.randint(0, 50)
            }
        ]
    }

def generate_diff():
    return f"""diff --git a/{generate_random_string(10)}.py b/{generate_random_string(10)}.py
index 1234567..abcdef0 100644
--- a/{generate_random_string(10)}.py
+++ b/{generate_random_string(10)}.py
@@ -1,5 +1,5 @@
 def old_function():
-    return "old"
+    return "new"
"""

def generate_comments(count=3):
    return [f"Comment {i+1}: {generate_random_string(50)}" for i in range(count)]

def generate_sample(sample_id):
    issue_title = f"Issue #{sample_id}: {generate_random_string(30)}"
    issue_body = f"Description of issue {sample_id}: {generate_random_string(100)}"
    issue_comments = generate_comments(random.randint(1, 5))

    pr_title = f"PR #{sample_id}: {generate_random_string(30)}"
    pr_body = f"Description of PR {sample_id}: {generate_random_string(100)}"
    pr_comments = generate_comments(random.randint(1, 5))

    commits = [generate_commit() for _ in range(random.randint(1, 3))]
    diffs = generate_diff()
    comments = generate_comments(random.randint(2, 6))

    text = f"""Issue:
{issue_title}
{issue_body}
Comments:
{chr(10).join(issue_comments)}

PR:
{pr_title}
{pr_body}
Comments:
{chr(10).join(pr_comments)}

Commits:
{chr(10).join([f"Hash: {c['hash']}, Message: {c['message']}, Modifications: {c['modifications']}" for c in commits])}

Diffs:
{diffs}

Comments:
{chr(10).join(comments)}
"""
    return text

def main():
    output_dir = "data/samples"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 51):
        sample_id = f"{i:02d}"
        sample_data = generate_sample(i)
        filename = f"sample_{sample_id}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(sample_data)

        print(f"Generated {filename}")

    print(f"Generated 50 sample files in {output_dir}")

if __name__ == "__main__":
    main()