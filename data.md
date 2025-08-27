Perfect! I've cleaned up the failed extraction. Now let's use the **regular extraction method** which will definitely work and give you the actual code diffs:

## 🚀 **Run Working Extraction**

```bash
uv run data/extract.py \
  --repo https://github.com/juspay/hyperswitch.git \
  --hf-namespace archit11 \
  --hf-dataset hyperswitch-code-history \
  --workdir data/_work_hyperswitch \
  --create-pretraining-dataset \
  --max-workers 32 \
  --commit-timeout 120
```

## 📊 **What You'll Get This Time**

### **✅ Actual Code Content (Not Just Metadata):**

**1. Code Diffs - `patches/` directory:**
```bash
# View actual code diffs
zcat data/_work_hyperswitch/dataset/patches/XX/XXXX.patch.jsonl.gz | head -1 | jq .
```
**Output:**
```json
{
  "commit": "abc123...",
  "path": "src/main.rs", 
  "patch": "@@ -10,5 +10,7 @@\n-    let x = 1;\n+    let x = 42;\n+    println!(\"Hello!\");\n     let y = 2;"
}
```

**2. File Content - `snapshots/` directory:**
```bash
# View before/after file content
zcat data/_work_hyperswitch/dataset/snapshots/XX/XXXX/before/src__main__rs.gz | zcat
```

**3. Pretraining Dataset - `pretraining/` directory:**
```bash
# View combined dataset perfect for model training
zcat data/_work_hyperswitch/dataset/pretraining/pretraining-0000.jsonl.gz | head -1 | jq .
```
**Output:**
```json
{
  "commit_hash": "abc123...",
  "commit_message": "Add new feature with improved error handling",
  "changes": [
    {
      "file_path": "src/main.rs",
      "diff": "@@ -10,5 +10,7 @@\n-    let x = 1;\n+    let x = 42;\n+    println!(\"Hello!\");",
      "before_content": "    let x = 1;\n    let y = 2;",
      "after_content": "    let x = 42;\n    println!(\"Hello!\");\n    let y = 2;"
    }
  ]
}
```

## 🎯 **Key Difference**

**❌ What you were seeing (files_index.jsonl.gz):**
- Only file metadata: `path`, `size`, `hash`
- No actual code content

**✅ What you'll get now:**
- **Actual code diffs** with `+` and `-` lines
- **Complete file content** before and after changes
- **Pretraining-ready format** combining everything

The regular extraction method will work reliably and give you the actual code content you need for training models! 🎉
