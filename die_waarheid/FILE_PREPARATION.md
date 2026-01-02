# Die Waarheid - File Preparation Guide

## 📁 How to Prepare Your Files for Analysis

### 🎯 Quick Answer

**You DON'T need to zip anything!** Just give the program your actual file paths.

---

## 🚀 Easiest Method: Direct File Paths

### Step 1: Gather Your Files

Put your files anywhere on your computer:

- Desktop

- Documents folder

- Downloads

- Any folder

### Step 2: Run the Program

```bash
python interactive_analysis.py
```

### Step 3: Enter Full File Paths

When the program asks, type the **full path** to each file:

```text
📁 What files do you want to analyze?
File #1: C:\Users\YourName\Desktop\statement.txt
   ✅ Added: C:\Users\YourName\Desktop\statement.txt

File #2: C:\Users\YourName\Documents\interview.wav

   ✅ Added: C:\Users\YourName\Documents\interview.wav
File #3: [Press Enter to finish]
```

### How to Get File Paths

**Windows Method 1 - Copy Path:**

1. Right-click the file
2. Hold **Shift** + right-click
3. Select "Copy as path"
4. Paste into the program

**Windows Method 2 - Properties:**

1. Right-click file → Properties
2. Copy the "Location" path
3. Add the filename at the end

**Example:**

- Location: `C:\Users\YourName\Desktop`
- Filename: `statement.txt`
- Full path: `C:\Users\YourName\Desktop\statement.txt`

---

## 📂 What Files Can You Analyze?

### ✅ Text Files

- `.txt` - Plain text statements
- `.doc` - Word documents
- `.docx` - New Word documents

### ✅ Audio Files  

- `.wav` - Best quality
- `.mp3` - Common format
- `.m4a` - iPhone recordings
- `.flac` - High quality

### ✅ Chat Exports

- WhatsApp exports (any format)
- SMS exports
- Messenger exports
- Any text-based chat log

---

## Example Workflow

### Scenario: You have a witness statement and interview recording

### Files You Have

- `witness_statement.txt` (on Desktop)

- `interview_recording.wav` (in Documents)

### What You Do

```bash
python interactive_analysis.py
```

### What You Enter

```text
📁 What files do you want to analyze?
File #1: C:\Users\YourName\Desktop\witness_statement.txt
   ✅ Added: C:\Users\YourName\Desktop\witness_statement.txt
File #2: C:\Users\YourName\Documents\interview_recording.wav
   ✅ Added: C:\Users\YourName\Documents\interview_recording.wav
File #3: [Press Enter to finish]
```

### What Happens

- Program analyzes both files

- Finds contradictions between text and audio

- Gives you evidence scores

- Creates investigative checklist

---

## Common Mistakes to Avoid

### Don't Use Full Paths

```text
File #1: statement.txt
File #2: interview.wav
```

**Note:** Program can't find the file without the full path

### Don't Use Spaces in Folder Names

```text
File #1: C:\My Files\statement.txt
```

**Note:** Spaces in folder names can cause issues

### Don't Add Quotes

```text
File #1: "C:\Users\Name\Desktop\statement.txt"
```

**Note:** Don't add quotes around the path

---

## Do This Instead

### Use Full Paths

```text
File #1: C:\Users\YourName\Desktop\statement.txt
File #2: C:\Users\YourName\Documents\interview.wav
```

### Test One File First

Start with just one file to make sure it works:

```text
📁 What files do you want to analyze?
File #1: C:\Users\YourName\Desktop\statement.txt
   ✅ Added: C:\Users\YourName\Desktop\statement.txt
File #2: [Press Enter to finish]
```

### Use Simple Folder Names

Put files in simple locations:

- `C:\Analysis\statement.txt`

- `C:\Analysis\interview.wav`

---

## Still Confused? Try This Test

### Create a Test File

1. Open Notepad
2. Type: "This is a test statement for analysis."
3. Save as: `C:\test_statement.txt`

### Run the Test

```bash
python interactive_analysis.py
```

### Enter the Test File

```text
📁 What files do you want to analyze?
File #1: C:\test_statement.txt
   ✅ Added: C:\test_statement.txt
File #2: [Press Enter to finish]
```

If this works, you're ready to analyze your real files!

---

## Need Help?

### File Path Issues

- Use "Copy as path" (Shift + right-click)
- Avoid spaces in folder names
- Use full paths starting with `C:\`

### File Format Issues

- Text: Save as `.txt` if unsure
- Audio: Convert to `.wav` for best results
- Chat: Export as plain text

### Program Errors

- Check file paths are correct
- Make sure files actually exist
- Start with one file first

---

## Bottom Line

**NO ZIP FILES NEEDED!** Just give the program the full paths to your actual files.

```bash
python interactive_analysis.py
```

Then enter paths like:

- `C:\Users\YourName\Desktop\statement.txt`
- `C:\Users\YourName\Documents\interview.wav`

That's it! 🎉
