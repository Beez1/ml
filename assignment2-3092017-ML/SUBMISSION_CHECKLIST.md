# Submission Checklist

Before submitting the assignment, please verify the following:

## Documentation
- [ ] Task1_LiteratureReview.txt has been converted to PDF
- [ ] Task2_CNN_Report.txt has been converted to PDF
- [ ] References in both documents are properly formatted (APA style)
- [ ] Tables and figures are referenced correctly in both documents
- [ ] Documents meet page requirements (Task 1: 2-3 pages, Task 2: up to 10 pages)

## Notebook
- [ ] Task2_CNN_Notebook.ipynb runs from start to finish without errors
- [ ] All cells have been executed with outputs visible
- [ ] Model architecture matches the description in the report
- [ ] Data preprocessing steps are clear and well-documented
- [ ] Evaluation metrics (accuracy, precision, recall, F1) are computed
- [ ] Visualization cells create clear, labeled figures
- [ ] Model is saved to saved_model/ directory

## Final Submission Files
- [ ] final_submission/Task1_LiteratureReview.pdf
- [ ] final_submission/Task2_CNN_Report.pdf
- [ ] notebook/Task2_CNN_Notebook.ipynb (with outputs)
- [ ] saved_model/ (containing trained model after notebook execution)
- [ ] requirements.txt

## Additional Checks
- [ ] No large data files are included in the submission (only code and documentation)
- [ ] All references to figures in reports match the actual figures in the notebook
- [ ] README.md is up-to-date with correct instructions
- [ ] All file paths in the notebook are relative and will work when executed

## Submission Format
- [ ] All files are organized as required by the assignment specification
- [ ] The total size of the submission package is reasonable (excluding the dataset)

## Final Steps
1. Convert text documents to PDF:
   ```bash
   # Using pandoc (if installed)
   pandoc doc/Task1_LiteratureReview.txt -o final_submission/Task1_LiteratureReview.pdf
   pandoc doc/Task2_CNN_Report.txt -o final_submission/Task2_CNN_Report.pdf
   
   # Alternatively, open in a word processor and save as PDF
   ```

2. Execute the notebook one final time:
   ```bash
   jupyter notebook notebook/Task2_CNN_Notebook.ipynb
   ```

3. Create final submission archive:
   ```bash
   # Create submission zip file (excluding dataset)
   zip -r assignment2_submission.zip README.md final_submission/ notebook/ requirements.txt saved_model/ -x "*.DS_Store" -x "*__pycache__*"
   ```

4. Submit the zip file through the designated submission portal by Sunday 27/04 at 23:59. 